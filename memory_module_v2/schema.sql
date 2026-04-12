CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS memory_v2;

CREATE TABLE IF NOT EXISTS memory_v2.memory_exchanges (
  exchange_id text PRIMARY KEY,
  session_id text NOT NULL,
  ply_start integer NOT NULL,
  ply_end integer NOT NULL,
  verbatim_text text NOT NULL,
  verbatim_snippet text NOT NULL,
  message_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_exchanges_session_ply
  ON memory_v2.memory_exchanges(session_id, ply_start, ply_end);

CREATE INDEX IF NOT EXISTS ix_memory_exchanges_session_id
  ON memory_v2.memory_exchanges(session_id);

CREATE TABLE IF NOT EXISTS memory_v2.memory_objects (
  object_id text PRIMARY KEY,
  exchange_id text NOT NULL REFERENCES memory_v2.memory_exchanges(exchange_id) ON DELETE CASCADE,
  session_id text NOT NULL,
  ply_start integer NOT NULL,
  ply_end integer NOT NULL,
  exchange_core text NOT NULL,
  specific_context text NOT NULL,
  distill_text text NOT NULL,
  route text NOT NULL DEFAULT '',
  intent text NOT NULL DEFAULT '',
  function_name text NOT NULL DEFAULT '',
  slots jsonb NOT NULL DEFAULT '{}'::jsonb,
  keywords jsonb NOT NULL DEFAULT '[]'::jsonb,
  distill_model text NOT NULL DEFAULT '',
  distilled_at timestamptz NOT NULL DEFAULT now(),
  embedding vector(1024)
);

CREATE INDEX IF NOT EXISTS ix_memory_objects_session_id
  ON memory_v2.memory_objects(session_id);
