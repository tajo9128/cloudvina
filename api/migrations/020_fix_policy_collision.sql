-- FIX FOR ERROR 42710 (Policy already exists)
-- Run this in Supabase SQL Editor

-- 1. First, safely drop the existing policy to avoid conflicts
DROP POLICY IF EXISTS "Admins can view audit logs" ON public.fda_audit_logs;

-- 2. Re-create the policy with your email
-- REPLACE 'your_email@example.com' with your actual account email
CREATE POLICY "Admins can view audit logs"
    ON public.fda_audit_logs
    FOR SELECT
    TO authenticated
    USING (auth.jwt() ->> 'email' = 'cloudvina2025@gmail.com'); 

-- 3. Verify the table is working
SELECT count(*) as total_logs FROM public.fda_audit_logs;
