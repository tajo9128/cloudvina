export const API_BASE_URL = import.meta.env.VITE_AI_API_URL || "https://tajo9128-biodockify-ai.hf.space";
export const MAIN_PLATFORM_URL = "https://biodockify.com"; // Link back to main app

// Supabase Configuration (Shared)
// Reading from Env Vars or using defaults matching the main web app
export const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || 'https://ohzfktmtwmubyhvspexv.supabase.co';
export const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oemZrdG10d211YnlodnNwZXh2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM4MTk2MjgsImV4cCI6MjA3OTM5NTYyOH0.v8qHeRx5jkL8iaNEbEP_NMIvvUk4oidwwW6PkXo_DVY';
