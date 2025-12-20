-- RBAC System Tables
-- Prefix: rbac_ (Isolated)

-- 1. Roles Definition
CREATE TABLE IF NOT EXISTS public.rbac_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT UNIQUE NOT NULL, -- e.g., 'auditor'
    name TEXT NOT NULL, -- e.g., 'Compliance Auditor'
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Permissions Definition
CREATE TABLE IF NOT EXISTS public.rbac_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT UNIQUE NOT NULL, -- e.g., 'view_audit_logs'
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Role-Permission Mapping (Many-to-Many)
CREATE TABLE IF NOT EXISTS public.rbac_role_permissions (
    role_id UUID REFERENCES public.rbac_roles(id) ON DELETE CASCADE,
    permission_id UUID REFERENCES public.rbac_permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

-- 4. User-Role Mapping (Many-to-Many)
CREATE TABLE IF NOT EXISTS public.rbac_user_roles (
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES public.rbac_roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assigned_by UUID, -- Audit: who gave this role?
    PRIMARY KEY (user_id, role_id)
);

-- RLS Policies
ALTER TABLE public.rbac_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rbac_permissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rbac_role_permissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rbac_user_roles ENABLE ROW LEVEL SECURITY;

-- Read Access: Authenticated users can read definitions (to check their own access)
CREATE POLICY "Read Roles" ON public.rbac_roles FOR SELECT TO authenticated USING (true);
CREATE POLICY "Read Permissions" ON public.rbac_permissions FOR SELECT TO authenticated USING (true);
CREATE POLICY "Read Role Permissions" ON public.rbac_role_permissions FOR SELECT TO authenticated USING (true);
CREATE POLICY "Read User Roles" ON public.rbac_user_roles FOR SELECT TO authenticated USING (true);

-- Write Access: Only Admins (Profile.is_admin=TRUE)
CREATE POLICY "Admin Manage Roles" ON public.rbac_roles FOR ALL TO authenticated USING (
    EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND is_admin = TRUE)
);
CREATE POLICY "Admin Manage User Roles" ON public.rbac_user_roles FOR ALL TO authenticated USING (
    EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND is_admin = TRUE)
);


-- SEED DEFAULT DATA
INSERT INTO public.rbac_roles (code, name, description)
VALUES 
    ('auditor', 'Auditor', 'Can view audit logs and compliance reports'),
    ('manager', 'Manager', 'Can manage users and system settings'),
    ('researcher', 'Researcher', 'Standard user with advanced job capabilities')
ON CONFLICT (code) DO NOTHING;

INSERT INTO public.rbac_permissions (code, description)
VALUES 
    ('view_audit_logs', 'Access to FDA Compliance Logs'),
    ('manage_users', 'Create, Edit, Suspend Users'),
    ('view_system_config', 'View System Configuration')
ON CONFLICT (code) DO NOTHING;

-- Map Auditor -> view_audit_logs
DO $$
DECLARE
    r_auditor UUID;
    p_audit_view UUID;
BEGIN
    SELECT id INTO r_auditor FROM public.rbac_roles WHERE code = 'auditor';
    SELECT id INTO p_audit_view FROM public.rbac_permissions WHERE code = 'view_audit_logs';
    
    IF r_auditor IS NOT NULL AND p_audit_view IS NOT NULL THEN
        INSERT INTO public.rbac_role_permissions (role_id, permission_id)
        VALUES (r_auditor, p_audit_view)
        ON CONFLICT DO NOTHING;
    END IF;
END $$;
