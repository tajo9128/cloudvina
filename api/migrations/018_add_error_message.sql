-- Migration 018: Add Error Message Column to Jobs
-- Stores the error reason from AWS Batch when a docking job fails

-- Add error_message column for storing failure details
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS error_message TEXT;

-- Add comment for documentation
COMMENT ON COLUMN jobs.error_message IS 'Error message/reason from AWS Batch when job fails';

-- Reload schema cache
NOTIFY pgrst, 'reload config';
