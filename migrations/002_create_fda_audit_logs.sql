-- FDA 21 CFR Part 11 Compliance: Immutable Audit Logs
-- Updated based on user request

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS public.fda_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id),
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}'::jsonb,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    signature TEXT -- For tamper-evident hashing
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_user ON public.fda_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_resource ON public.fda_audit_logs(resource_id);
CREATE INDEX IF NOT EXISTS idx_fda_audit_logs_date ON public.fda_audit_logs(created_at);

-- RLS Policies (Immutable Logs)
ALTER TABLE public.fda_audit_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins can view audit logs" ON public.fda_audit_logs;

CREATE POLICY "Admins can view audit logs"
    ON public.fda_audit_logs
    FOR SELECT
    TO authenticated
    USING (auth.jwt() ->> 'email' IN ('REPLACE_WITH_YOUR_EMAIL'));

DROP POLICY IF EXISTS "Users can insert audit logs" ON public.fda_audit_logs;

CREATE POLICY "Users can insert audit logs"
    ON public.fda_audit_logs
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);
