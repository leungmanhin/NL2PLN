"""
SIMBA with per-batch checkpoint save.

DSPy's SIMBA.compile() is monolithic and exposes no per-batch hook.
This subclass overrides compile() with a verbatim copy of DSPy 3.1.3's
SIMBA.compile() body plus a save call at each batch boundary, so a
Colab SIGKILL leaves the latest best-on-mini-batch program saved to
disk (recoverable; the existing try/finally in simba.py only saves on
Python exceptions, not on SIGKILL).

The saved checkpoint is the candidate that won the most recent mini-
batch evaluation (winning_programs[-1] in SIMBA's terminology).  This
is an approximation of "best so far" — it's based on a single 16-example
(or whatever bsize) mini-batch, not the full trainset.  After a clean
full run, the VALIDATION pass at compile()'s end produces a different
"best on full trainset" program which the caller's existing finally-save
writes to --output.  So in success-case the checkpoint and output may
differ; in failure-case the checkpoint is what you have.

If you upgrade DSPy and SIMBA's internals change, refresh the body of
compile() below to match the new SIMBA source.  Pinned to DSPy 3.1.3.
"""
import logging
import random
from pathlib import Path

import dspy
import numpy as np
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.simba_utils import prepare_models_for_resampling, wrap_program

# Use the same logger name as SIMBA so checkpoint info appears alongside its
# own log lines (helpful in Colab where logs are interleaved).
logger = logging.getLogger("dspy.teleprompt.simba")


class CheckpointingSIMBA(SIMBA):
    """SIMBA with per-batch checkpoint save.

    Identical to dspy.teleprompt.SIMBA except that after each batch's
    winner is selected (STEP 7), the winner is saved to
    ``checkpoint_path`` via ``program.save()``.  Subsequent batches
    overwrite the file.

    Args:
        checkpoint_path: file path; if None, behaves exactly like SIMBA.
        All other args forwarded to SIMBA.__init__.
    """

    def __init__(self, *args, checkpoint_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self._checkpoint_path is not None:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, program, batch_idx: int) -> None:
        if self._checkpoint_path is None:
            return
        try:
            program.save(str(self._checkpoint_path))
            logger.info(
                f"CHECKPOINT: saved best-of-batch-{batch_idx + 1} to "
                f"{self._checkpoint_path}"
            )
        except Exception as e:
            logger.warning(f"CHECKPOINT save failed: {type(e).__name__}: {e}")

    def compile(self, student, *, trainset, seed=0):  # noqa: C901
        """Verbatim copy of DSPy 3.1.3's SIMBA.compile() body, plus a
        ``self._save_checkpoint`` call after STEP 7.  Marker:
        ``# CHECKPOINT POINT`` flags the inserted line.
        """
        # ----- Basic checks -----
        assert len(trainset) >= self.bsize, (
            f"Trainset too small: {len(trainset)} < {self.bsize}"
        )

        # ----- Initialize RNG -----
        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        programs = []
        program_scores = {}
        next_program_idx = 0

        # ----- Helper functions (closure over local state) -----
        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        def top_k_plus_baseline(k: int) -> list[int]:
            scored_programs = sorted(
                programs, key=lambda p: calc_average_score(p.simba_idx), reverse=True
            )
            top_k = [p.simba_idx for p in scored_programs[:k]]
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def softmax_sample(rng_obj, program_idxs, temperature):
            if not program_idxs:
                raise ValueError("No programs available for softmax sampling.")
            scores = [calc_average_score(idx) for idx in program_idxs]
            exps = [np.exp(s / temperature) for s in scores]
            sum_exps = sum(exps)
            if sum_exps <= 0:
                return rng_obj.choice(program_idxs)
            probs = [val / sum_exps for val in exps]
            return rng_obj.choices(program_idxs, weights=probs, k=1)[0]

        def register_new_program(prog, score_list):
            nonlocal next_program_idx
            next_program_idx += 1
            new_idx = next_program_idx
            prog.simba_idx = new_idx
            programs.append(prog)
            program_scores[new_idx] = score_list

        # ----- Initialize the baseline program: index=0 -----
        student = student.deepcopy()
        student.simba_idx = 0
        programs.append(student)
        program_scores[0] = []

        winning_programs = [student]

        # ----- Data shuffling -----
        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0

        # ----- Parallel runner -----
        run_parallel = dspy.Parallel(
            access_examples=False, num_threads=self.num_threads
        )

        trial_logs = {}
        for batch_idx in range(self.max_steps):
            trial_logs[batch_idx] = {}

            logger.info(f"Starting batch {batch_idx + 1} of {self.max_steps}.")

            # STEP 1: Get next batch
            if instance_idx + self.bsize > len(trainset):
                rng.shuffle(data_indices)
                instance_idx = 0
            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize

            models = prepare_models_for_resampling(
                programs[0], self.num_candidates, self.teacher_settings
            )
            top_programs = top_k_plus_baseline(self.num_candidates)

            exec_pairs = []
            predictor2name = {}

            for model in models:
                for example in batch:
                    chosen_prog_idx = softmax_sample(
                        rng, top_programs, self.temperature_for_sampling
                    )
                    candidate_system = programs[chosen_prog_idx].deepcopy()
                    candidate_system.set_lm(model)
                    for name, predictor in candidate_system.named_predictors():
                        predictor2name[id(predictor)] = name
                    wrapped_candidate_system = wrap_program(
                        candidate_system, self.metric
                    )
                    exec_pairs.append((wrapped_candidate_system, example))

            # STEP 2: Execute trajectory sampling
            logger.info(
                f"Sampling program trajectories on {self.bsize} examples x "
                f"{self.num_candidates} samples."
            )
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(exec_pairs) == self.bsize * self.num_candidates

            # STEP 3: Sort buckets
            buckets = []
            largest_max_to_avg_gap = float("-inf")
            batch_10th_percentile_score = np.percentile(
                [float(o["score"]) for o in outputs], 10
            )
            batch_90th_percentile_score = np.percentile(
                [float(o["score"]) for o in outputs], 90
            )

            for idx, _ in enumerate(batch):
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)
                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                max_to_min_gap = max_score - min_score
                max_to_avg_gap = max_score - avg_score
                if max_to_avg_gap > largest_max_to_avg_gap:
                    largest_max_to_avg_gap = max_to_avg_gap
                buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))

            buckets.sort(key=lambda x: x[1], reverse=True)

            all_scores_in_this_batch = [o["score"] for o in outputs]
            baseline_score = sum(all_scores_in_this_batch) / len(
                all_scores_in_this_batch
            )
            logger.info(
                f"Batch {batch_idx + 1}: Baseline mini-batch score: {baseline_score}\n"
            )

            # STEP 4: Build new candidate programs
            system_candidates = []
            for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
                max_to_min_gap, max_score, max_to_avg_gap = bucket_stats
                logger.info(
                    f"Batch {batch_idx + 1}: Processing bucket #{bucket_idx + 1}, "
                    f"with max score {max_score}, max-to-min gap {max_to_min_gap}, "
                    f"and max-to-avg gap {max_to_avg_gap}."
                )

                src_prog_idx = softmax_sample(
                    rng,
                    top_k_plus_baseline(self.num_candidates),
                    self.temperature_for_candidates,
                )
                system_candidate = programs[src_prog_idx].deepcopy()

                name2predictor = {}
                num_demos_list = []
                max_demos_tmp = self.max_demos if self.max_demos > 0 else 3

                for name, predictor in system_candidate.named_predictors():
                    name2predictor[name] = predictor
                    num_demos_list.append(len(predictor.demos))

                num_demos = max(num_demos_list) if num_demos_list else 0
                num_demos_to_drop = max(
                    rng_np.poisson(num_demos / max_demos_tmp),
                    int(num_demos >= max_demos_tmp),
                )
                num_demos_to_drop = min(num_demos_to_drop, num_demos)
                demos_to_drop = [
                    rng.randrange(num_demos) for _ in range(num_demos_to_drop)
                ]

                for _, predictor in name2predictor.items():
                    predictor.demos = [
                        demo
                        for idxd, demo in enumerate(predictor.demos)
                        if idxd not in demos_to_drop
                    ]

                strategy = rng.choice(self.strategies)
                logger.info(
                    f"Batch {batch_idx + 1}: Invoking strategy: {strategy.__name__}"
                    + (
                        f", having dropped {num_demos_to_drop} demos per predictor"
                        if num_demos_to_drop
                        else ""
                    )
                )

                try:
                    strategy(
                        bucket,
                        system_candidate,
                        predictor2name=predictor2name,
                        name2predictor=name2predictor,
                        batch_10p_score=batch_10th_percentile_score,
                        batch_90p_score=batch_90th_percentile_score,
                        prompt_model=self.prompt_model,
                    )
                except Exception as e:
                    logger.error(f"Strategy failed with error: {e}")
                    continue

                system_candidates.append(system_candidate)
                logger.info("\n")

                if len(system_candidates) >= self.num_candidates + 1:
                    break

            # STEP 5: Evaluate new system_candidates on the same mini-batch
            logger.info(
                f"Batch {batch_idx + 1}: Evaluating {len(system_candidates)} programs "
                f"on {self.bsize} examples."
            )

            exec_pairs = [
                (wrap_program(sys, self.metric), ex)
                for sys in system_candidates
                for ex in batch
            ]
            outputs = run_parallel(exec_pairs)
            assert (
                len(outputs)
                == len(exec_pairs)
                == len(system_candidates) * self.bsize
            )

            # STEP 6: Compute average mini-batch scores per candidate
            candidate_scores = []
            for idx_cand, _ in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                avg_sys_score = sum(sys_scores) / len(sys_scores)
                candidate_scores.append(avg_sys_score)

            logger.info(
                f"Scores after {batch_idx + 1} batches: {candidate_scores}, "
                f"Best: {max(candidate_scores) if candidate_scores else 'N/A'}\n"
            )

            # STEP 7: Select the best for "winning" record
            if candidate_scores:
                best_idx_among_candidates = candidate_scores.index(
                    max(candidate_scores)
                )
                best_program = system_candidates[best_idx_among_candidates]
                winning_programs.append(best_program.deepcopy())

                # CHECKPOINT POINT: save best-of-mini-batch after each batch
                self._save_checkpoint(best_program, batch_idx)

            # STEP 8: Register all new candidate systems in our global pool
            for idx_cand, cand_sys in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                register_new_program(cand_sys, sys_scores)

        # ----- Validation pass on full trainset -----
        M = len(winning_programs) - 1  # noqa: N806
        N = self.num_candidates + 1  # noqa: N806
        if M < 1:
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]

        program_idxs = list(dict.fromkeys(program_idxs))

        candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]
        logger.info(
            f"VALIDATION: Evaluating {len(candidate_programs)} programs "
            f"on the full trainset."
        )
        exec_pairs = [
            (wrap_program(sys, self.metric), ex)
            for sys in candidate_programs
            for ex in trainset
        ]
        outputs = run_parallel(exec_pairs)

        scores = []
        for idx_prog, _ in enumerate(candidate_programs):
            start = idx_prog * len(trainset)
            end = (idx_prog + 1) * len(trainset)
            sys_scores = [outputs[i]["score"] for i in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)
            if idx_prog != 0:
                trial_logs[idx_prog - 1]["train_score"] = avg_score

        assert len(scores) == len(candidate_programs)
        candidate_data = [
            {"score": s, "program": p}
            for s, p in zip(scores, candidate_programs, strict=False)
        ]
        candidate_data.sort(key=lambda x: x["score"], reverse=True)

        best_idx = scores.index(max(scores)) if scores else 0
        best_program = candidate_programs[best_idx].deepcopy()
        logger.info(
            f"Final trainset scores: {scores}, "
            f"Best: {max(scores) if scores else 'N/A'} "
            f"(at index {best_idx if scores else 'N/A'})\n\n\n"
        )

        best_program.candidate_programs = candidate_data
        best_program.trial_logs = trial_logs

        return best_program
