-- FIX: Add missing INSERT/UPDATE policies for daily_job_usage
-- This fixes the "new row violates row-level security policy" error

-- 1. Allow users to INSERT their own usage records
DROP POLICY IF EXISTS "Users can insert own usage" ON daily_job_usage;
CREATE POLICY "Users can insert own usage" ON daily_job_usage
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 2. Allow users to UPDATE their own usage records
DROP POLICY IF EXISTS "Users can update own usage" ON daily_job_usage;
CREATE POLICY "Users can update own usage" ON daily_job_usage
    FOR UPDATE USING (auth.uid() = user_id);

-- 3. Ensure SELECT is also correct (already exists but good to be safe)
DROP POLICY IF EXISTS "Users can view own usage" ON daily_job_usage;
CREATE POLICY "Users can view own usage" ON daily_job_usage
    FOR SELECT USING (auth.uid() = user_id);

-- 4. Reload schema cache to apply changes immediately
NOTIFY pgrst, 'reload config';
