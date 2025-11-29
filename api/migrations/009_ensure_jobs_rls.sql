-- FIX: Ensure RLS policies for JOBS table are correct
-- Run this if you encounter RLS errors for the 'jobs' table

-- 1. Enable RLS
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

-- 2. Allow users to INSERT their own jobs
DROP POLICY IF EXISTS "Users can insert own jobs" ON jobs;
CREATE POLICY "Users can insert own jobs" ON jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 3. Allow users to VIEW their own jobs
DROP POLICY IF EXISTS "Users can view own jobs" ON jobs;
CREATE POLICY "Users can view own jobs" ON jobs
    FOR SELECT USING (auth.uid() = user_id);

-- 4. Allow users to UPDATE their own jobs (e.g. status)
DROP POLICY IF EXISTS "Users can update own jobs" ON jobs;
CREATE POLICY "Users can update own jobs" ON jobs
    FOR UPDATE USING (auth.uid() = user_id);

-- 5. Grant permissions
GRANT ALL ON public.jobs TO postgres;
GRANT ALL ON public.jobs TO service_role;
GRANT ALL ON public.jobs TO authenticated;

NOTIFY pgrst, 'reload config';
