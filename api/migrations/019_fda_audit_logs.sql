-- FDA 21 CFR Part 11 Audit Trail Table
-- Prefix: fda_ (Isolated specific for compliance)

CREATE TABLE IF NOT EXISTS public.fda_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID, -- Nullable for system events, usually references auth.users
    action TEXT NOT NULL, -- e.g., 'JOB_SUBMISSION', 'REPORT_EXPORT', 'LOGIN'
    resource_id TEXT, -- ID of the object being acted upon (job_id, file_id)
    details JSONB DEFAULT '{}'::jsonb, -- Flex field for metadata (old_value, new_value)
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    signature TEXT -- Placeholder for digital signature/checksum
);

-- Index for fast searching by Resource or User
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_user ON public.fda_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_resource ON public.fda_audit_logs(resource_id);
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_date ON public.fda_audit_logs(created_at);

-- RLS Security
ALTER TABLE public.fda_audit_logs ENABLE ROW LEVEL SECURITY;

-- 1. Everyone can INSERT (System logs actions for everyone)
CREATE POLICY "Enable insert for authenticated users"
    ON public.fda_audit_logs
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- 2. Only Admins can VIEW (Regular users cannot verify logs, only auditors)
-- Assuming a 'profiles' table with 'is_admin' or similar logic. 
-- For safety/mvp, we allow users to see THEIR OWN logs, and Admins to see ALL.
CREATE POLICY "Users can view their own audit logs"
    ON public.fda_audit_logs
    FOR SELECT
    TO authenticated
    USING (
        auth.uid() = user_id 
        OR 
        EXISTS (
            SELECT 1 FROM public.profiles 
            WHERE profiles.id = auth.uid() 
            AND (profiles.designation ILIKE '%admin%' OR profiles.designation ILIKE '%manager%') -- Weak admin check fallback
        )
    );

-- 3. NO UPDATE/DELETE allowed (Immutable)
-- No policies created for UPDATE or DELETE implies they are forbidden by default RLS (Deny All).
