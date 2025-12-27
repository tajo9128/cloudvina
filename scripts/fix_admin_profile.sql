-- 1. Check if the user exists in the Auth system
SELECT id, email, created_at, last_sign_in_at 
FROM auth.users 
WHERE email = 'biodockify@hotmail.com';

-- 2. IF the user exists in auth.users but NOT in profiles, force insert them:
INSERT INTO public.profiles (id, email, role, is_admin, credits, plan)
SELECT 
    id, 
    email, 
    'admin', 
    true, 
    1000, 
    'premium'
FROM auth.users
WHERE email = 'biodockify@hotmail.com'
ON CONFLICT (id) DO UPDATE
SET 
    is_admin = true,
    role = 'admin';

-- 3. Verify final state
SELECT * FROM public.profiles WHERE email = 'biodockify@hotmail.com';
