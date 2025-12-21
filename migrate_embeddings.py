#!/usr/bin/env python3
"""
Embedding Migration Script

This script migrates existing embeddings from the old SentenceTransformer model
(all-MiniLM-L6-v2, 384 dimensions) to the new Gemini Embedding 001 model
(768 dimensions via OpenRouter API).

Usage:
    python migrate_embeddings.py [--batch-size N] [--dry-run]

Requirements:
    - OPENROUTER_API_KEY must be set in environment or .env file
    - Database must have been migrated to VECTOR(768) column

Notes:
    - This script processes embeddings in batches to handle rate limits
    - Progress is committed after each batch to allow resuming if interrupted
    - Use --dry-run to test without making changes
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from embeddings import compute_embedding_for_document, EMBEDDING_DIMENSIONS, EmbeddingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_connection():
    """Create a direct database connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    return psycopg2.connect(database_url, cursor_factory=RealDictCursor)


def get_entries_to_migrate(conn, limit=None, include_existing=False):
    """Get prompt_completion entries that need migration.

    Args:
        conn: Database connection
        limit: Optional limit on number of entries
        include_existing: If True, re-embed all entries; if False, only NULL embeddings
    """
    with conn.cursor() as cur:
        if include_existing:
            query = """
                SELECT id, prompt, completion
                FROM prompt_completion
                ORDER BY id
            """
        else:
            # Only get entries with NULL embeddings (for resumability)
            query = """
                SELECT id, prompt, completion
                FROM prompt_completion
                WHERE embedding IS NULL
                ORDER BY id
            """
        if limit:
            query += f" LIMIT {limit}"
        cur.execute(query)
        return cur.fetchall()


def create_vector_index(conn):
    """Create the IVFFlat index after embeddings are populated."""
    logger.info("Creating IVFFlat index on embeddings...")
    with conn.cursor() as cur:
        # Drop existing index if any
        cur.execute("DROP INDEX IF EXISTS prompt_completion_embedding_idx")
        # Create new index with IVFFlat
        cur.execute("""
            CREATE INDEX prompt_completion_embedding_idx
                ON prompt_completion
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
        """)
    conn.commit()
    logger.info("Index created successfully")


def update_embedding(conn, entry_id, embedding):
    """Update a single entry's embedding."""
    with conn.cursor() as cur:
        # Convert numpy array to list for PostgreSQL
        embedding_list = embedding.tolist()
        cur.execute(
            """
            UPDATE prompt_completion
            SET embedding = %s::vector
            WHERE id = %s
            """,
            (str(embedding_list), entry_id)
        )


def migrate_embeddings(batch_size=10, dry_run=False, limit=None, recompute_all=False, create_index=False):
    """
    Migrate all embeddings to the new Gemini model.

    Args:
        batch_size: Number of entries to process before committing
        dry_run: If True, don't actually update the database
        limit: Optional limit on number of entries to process
        recompute_all: If True, re-embed all entries; if False, only NULL embeddings
        create_index: If True, create the IVFFlat index after migration
    """
    logger.info("=" * 60)
    logger.info("Embedding Migration Script")
    logger.info("=" * 60)
    logger.info(f"Target dimensions: {EMBEDDING_DIMENSIONS}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Recompute all: {recompute_all}")
    logger.info(f"Create index: {create_index}")
    if limit:
        logger.info(f"Limit: {limit}")
    logger.info("=" * 60)

    # Verify API key is set
    if not os.environ.get('OPENROUTER_API_KEY'):
        logger.error("OPENROUTER_API_KEY not set. Cannot proceed.")
        sys.exit(1)

    conn = get_database_connection()

    try:
        entries = get_entries_to_migrate(conn, limit, include_existing=recompute_all)
        total = len(entries)
        logger.info(f"Found {total} entries to migrate")

        if total == 0:
            logger.info("No entries to migrate. Done.")
            return

        success_count = 0
        error_count = 0
        start_time = datetime.now()

        for i, entry in enumerate(entries):
            entry_id = entry['id']
            prompt = entry['prompt']
            completion = entry['completion']

            try:
                # Format as Q&A for better embedding quality
                combined_text = f"Question: {prompt} Answer: {completion}"

                if dry_run:
                    logger.debug(f"[DRY RUN] Would embed entry {entry_id}: {combined_text[:50]}...")
                else:
                    embedding = compute_embedding_for_document(combined_text)
                    update_embedding(conn, entry_id, embedding)

                success_count += 1

                # Commit after each batch
                if not dry_run and (i + 1) % batch_size == 0:
                    conn.commit()
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = success_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) - "
                        f"Rate: {rate:.2f}/sec - "
                        f"Errors: {error_count}"
                    )

            except EmbeddingError as e:
                error_count += 1
                logger.error(f"Failed to embed entry {entry_id}: {e}")
                # Continue with next entry

            except Exception as e:
                error_count += 1
                logger.error(f"Unexpected error for entry {entry_id}: {e}")
                # Continue with next entry

            # Small delay to avoid rate limiting
            if not dry_run and (i + 1) % batch_size == 0:
                time.sleep(0.5)

        # Final commit
        if not dry_run:
            conn.commit()

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("Migration Complete")
        logger.info("=" * 60)
        logger.info(f"Total entries: {total}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Time elapsed: {elapsed:.1f} seconds")
        if elapsed > 0:
            logger.info(f"Average rate: {success_count / elapsed:.2f} entries/second")

        # Create index if requested
        if create_index and not dry_run and success_count > 0:
            create_vector_index(conn)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate embeddings to Gemini Embedding 001"
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=10,
        help='Number of entries to process before committing (default: 10)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Test run without making changes'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of entries to process (for testing)'
    )
    parser.add_argument(
        '--recompute-all', '-a',
        action='store_true',
        help='Recompute all embeddings, not just NULL ones'
    )
    parser.add_argument(
        '--create-index', '-i',
        action='store_true',
        help='Create IVFFlat index after migration completes'
    )

    args = parser.parse_args()

    migrate_embeddings(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        limit=args.limit,
        recompute_all=args.recompute_all,
        create_index=args.create_index
    )


if __name__ == '__main__':
    main()
