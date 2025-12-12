-- Fix: RLS Disabled in Public (public.pricing_plans)
-- Detected by Supabase Linter

-- 1. Enable Row Level Security
ALTER TABLE public.pricing_plans ENABLE ROW LEVEL SECURITY;

-- 2. Allow public read access (Pricing is public information)
DROP POLICY IF EXISTS "Public read access" ON public.pricing_plans;
CREATE POLICY "Public read access"
ON public.pricing_plans
FOR SELECT
TO public
USING (true);

-- 3. Restrict write access to service_role only (Implicit in RLS, but for clarity)
-- No policy needed for service_role as it bypasses RLS by default.
-- Regular authenticated users should NOT be able to modify pricing.
