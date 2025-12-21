-- Ensure 'jobs' table exists for docking
-- This table tracks AWS Batch jobs

CREATE TABLE IF NOT EXISTS public.jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    batch_id UUID, -- Group ID for batch submissions
    status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, SUBMITTED, RUNNING, SUCCEEDED, FAILED
    receptor_s3_key TEXT,
    ligand_s3_key TEXT,
    receptor_filename VARCHAR(255),
    ligand_filename VARCHAR(255),
    result_s3_key TEXT,
    binding_affinity FLOAT,
    docking_score FLOAT, -- Alias/Legacy
    vina_score FLOAT, -- Alias/Legacy
    docking_results JSONB, -- Store full JSON output
    batch_job_id VARCHAR(255), -- AWS Batch Job ID
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- RLS Policies
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own jobs" ON public.jobs;
CREATE POLICY "Users can view own jobs"
    ON public.jobs
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own jobs" ON public.jobs;
CREATE POLICY "Users can insert own jobs"
    ON public.jobs
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own jobs" ON public.jobs;
CREATE POLICY "Users can update own jobs"
    ON public.jobs
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON public.jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON public.jobs(user_id);
