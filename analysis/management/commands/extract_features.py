import math
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction

from courses.models import Assignment
from submissions.models import Submission
from analysis.models import AnalysisRun, SubmissionFeatures

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_sentence_count(text: str) -> int:
    # quick heuristic: count ., ?, !
    if not text:
        return 0
    return max(1, sum(text.count(x) for x in [".", "?", "!"]))


def compute_text_features(text: str) -> dict:
    text = text or ""
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    char_count = len(text)
    sentence_count = simple_sentence_count(text)

    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    avg_sentence_len_words = (word_count / sentence_count) if sentence_count else 0.0

    # Type-token ratio
    lowered = [w.lower() for w in words]
    ttr = (len(set(lowered)) / word_count) if word_count else 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_len": round(avg_word_len, 4),
        "avg_sentence_len_words": round(avg_sentence_len_words, 4),
        "type_token_ratio": round(ttr, 4),
    }


class Command(BaseCommand):
    help = "Extract text features + embeddings for an assignment and store them in SubmissionFeatures under a new AnalysisRun."

    def add_arguments(self, parser):
        parser.add_argument("assignment_id", type=int)
        parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
        parser.add_argument("--store_embedding", action="store_true", help="Store embedding vector in DB (OK for small datasets).")

        # REQUIRED: who is running this analysis
        parser.add_argument(
        "--user_id",
        type=int,
        required=True,
        help="ID of the user creating this AnalysisRun (Admin or Researcher)"
    )

    def handle(self, *args, **options):
        assignment_id = options["assignment_id"]
        user_id = options["user_id"] 
        model_name = options["model"]
        store_embedding = options["store_embedding"]

        assignment = Assignment.objects.filter(id=assignment_id).first()
        if not assignment:
            raise CommandError(f"Assignment {assignment_id} not found.")

        submissions = list(
            Submission.objects.filter(assignment_id=assignment_id)
            .select_related("student")
            .order_by("created_at")
        )
        if not submissions:
            self.stdout.write(self.style.WARNING("No submissions found. Nothing to do."))
            return

        self.stdout.write(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        texts = [s.text or "" for s in submissions]

        self.stdout.write(f"Computing embeddings for {len(texts)} submissions...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # centroid similarity (each vector vs mean vector)
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid)

        sims = cosine_similarity(embeddings, centroid).reshape(-1)

        config = {
            "embedding_model": model_name,
            "features": ["basic_text_stats", "centroid_similarity"],
            "store_embedding": bool(store_embedding),
        }

        with transaction.atomic():
            run = AnalysisRun.objects.create(
                assignment_id=assignment_id,
                created_by_id=user_id,  
                config=config,
                status=AnalysisRun.Status.RUNNING,
                started_at=timezone.now(),
            )

            # Upsert features rows
            for s, emb, sim in zip(submissions, embeddings, sims):
                feats = compute_text_features(s.text)
                feats["centroid_similarity"] = float(sim)

                SubmissionFeatures.objects.update_or_create(
                    analysis_run=run,
                    submission=s,
                    defaults={
                        "features": feats,
                        "ai_similarity": None,
                        "predicted_ai_assist_prob": None,
                        "cluster_id": None,
                        "embedding": emb.tolist() if store_embedding else None,
                    },
                )

            run.status = AnalysisRun.Status.DONE
            run.finished_at = timezone.now()
            run.save()

        self.stdout.write(self.style.SUCCESS(f"Done. AnalysisRun={run.id}"))
        self.stdout.write("Next: export joined dataset (rubric targets + features).")
