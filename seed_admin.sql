-- Replace 'your-email@example.com' with your actual email
UPDATE profiles 
SET is_admin = TRUE 
WHERE id IN (
    SELECT id FROM auth.users WHERE email = 'your-email@example.com'
);

-- Verify the change
SELECT * FROM profiles WHERE is_admin = TRUE;
