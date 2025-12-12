-- Add batch_id to jobs table to support Batch Docking
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS batch_id UUID;

-- Add index for faster queries on batches
CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs(batch_id);
