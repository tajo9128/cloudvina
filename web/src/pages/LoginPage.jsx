import { useState } from 'react'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import { useNavigate, Link } from 'react-router-dom'

export default function LoginPage() {
    const [loading, setLoading] = useState(false)
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [designation, setDesignation] = useState('')
    const [organization, setOrganization] = useState('')
    const [isSignUp, setIsSignUp] = useState(false)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(null)
    const navigate = useNavigate()

    const handleAuth = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError(null)
        setSuccess(null)

        try {
            if (isSignUp) {
                // 1. Sign up with Supabase
                const { data: authData, error: authError } = await supabase.auth.signUp({
                    email,
                    password,
                })

                if (authError) throw authError

                if (authData.user) {
                    // 2. Create user profile in backend
                    const response = await fetch(`${API_URL}/auth/signup`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            email,
                            password,
                            designation,
                            organization
                        })
                    })

                    if (!response.ok) {
                        const data = await response.json()
                        throw new Error(data.detail || 'Signup failed')
                    }

                    // Check if session exists (email verification might be required)
                    if (authData.session) {
                        // User is logged in, navigate to dashboard
                        navigate('/dashboard')
                    } else {
                        // Email verification required
                        setSuccess('✅ Account created! Check your email to verify your account. Click the confirmation link once. Even if you see an error, try logging in - your account may be ready.')
                        setTimeout(() => {
                            setIsSignUp(false)
                            setEmail('')
                            setPassword('')
                            setDesignation('')
                            setOrganization('')
                        }, 5000)
                    }
                }
            } else {
                const { error } = await supabase.auth.signInWithPassword({
                    email,
                    password,
                })
                if (error) throw error
                navigate('/dashboard')
            }
        } catch (error) {
            // Handle specific error cases with user-friendly messages
            let errorMessage = error.message

            // Supabase rate limiting
            // Supabase rate limiting
            if (errorMessage.includes('For security purposes') || errorMessage.includes('request this after')) {
                console.warn("Supabase rate limit hit:", errorMessage)
                if (isSignUp) {
                    setSuccess('✅ Account created! Check your email to verify your account. Click the confirmation link once.')
                }
                // Suppress error for login as well, or handle differently if needed.
                // User explicitly asked to hide the "Please wait" message.
            }
            // Email already exists
            else if (errorMessage.includes('already registered') || errorMessage.includes('already exists')) {
                setError('📧 Email already registered. Check your Email to click Confirmation link, Click it once, You are Ready to login even if you get error in confirmation.')
            }
            // Password too weak
            else if (errorMessage.includes('password') && (errorMessage.includes('weak') || errorMessage.includes('short'))) {
                setError('🔒 Password must be at least 6 characters long.')
            }
            // Generic fallback
            else {
                setError(errorMessage)
            }
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-blue-mesh flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-2xl mb-4 shadow-lg shadow-cyan-500/30">
                        <span className="text-3xl">🧬</span>
                    </div>
                    <h2 className="text-3xl font-extrabold text-white tracking-tight">
                        {isSignUp ? 'Create your account' : 'Welcome back'}
                    </h2>
                    <p className="mt-2 text-sm text-blue-200">
                        {isSignUp ? 'Start your molecular docking journey' : 'Sign in to continue to BioDockify'}
                    </p>
                </div>

                {/* Form Card */}
                <div className="glass-modern p-8 rounded-2xl">
                    <form className="space-y-5" onSubmit={handleAuth}>
                        {/* Email */}
                        <div>
                            <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                Email Address *
                            </label>
                            <input
                                type="email"
                                required
                                className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                placeholder="you@university.edu"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                        </div>

                        {/* Password */}
                        <div>
                            <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                Password *
                            </label>
                            <input
                                type="password"
                                required
                                className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                            {isSignUp && (
                                <p className="mt-1 text-xs text-slate-500">Minimum 6 characters</p>
                            )}
                        </div>

                        {/* Signup-only fields */}
                        {isSignUp && (
                            <>
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Designation *
                                    </label>
                                    <select
                                        required
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        value={designation}
                                        onChange={(e) => setDesignation(e.target.value)}
                                    >
                                        <option value="" disabled>Select Designation</option>
                                        <option value="Student">Student (B.Tech/M.Tech/PhD)</option>
                                        <option value="Researcher">Researcher/Scientist</option>
                                        <option value="Professor">Professor/Faculty</option>
                                        <option value="Industry">Industry Professional</option>
                                        <option value="Other">Other</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Organization/University *
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        placeholder="e.g., IIT Delhi"
                                        value={organization}
                                        onChange={(e) => setOrganization(e.target.value)}
                                    />
                                </div>

                                {/* Info Box */}
                                <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-4">
                                    <div className="flex">
                                        <div className="flex-shrink-0">
                                            <svg className="h-5 w-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                            </svg>
                                        </div>
                                        <div className="ml-3">
                                            <p className="text-xs text-blue-800">
                                                <strong>Email verification required.</strong> Free users get 3 docking jobs per day. Upgrade anytime for unlimited access.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* Error/Success Messages */}
                        {error && (
                            <div className="bg-red-50 border border-red-200 rounded-xl p-3">
                                <p className="text-sm text-red-600">❌ {error}</p>
                            </div>
                        )}

                        {success && (
                            <div className="bg-green-50 border border-green-200 rounded-xl p-3">
                                <p className="text-sm text-green-600">{success}</p>
                            </div>
                        )}

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full btn-cyan py-3 rounded-xl font-bold text-lg flex justify-center items-center disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="flex items-center">
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Processing...
                                </span>
                            ) : (
                                isSignUp ? 'Create Account' : 'Sign In'
                            )}
                        </button>
                    </form>

                    {/* Toggle Sign Up/Sign In */}
                    <div className="mt-6 text-center">
                        <button
                            onClick={() => {
                                setIsSignUp(!isSignUp)
                                setError(null)
                                setSuccess(null)
                            }}
                            className="text-sm text-blue-600 hover:text-blue-700 font-bold hover:underline transition"
                        >
                            {isSignUp ? '← Already have an account? Sign in' : "Don't have an account? Sign up →"}
                        </button>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-8 text-center">
                    <Link to="/" className="text-sm text-blue-300 hover:text-cyan-400 transition font-medium">
                        ← Back to Homepage
                    </Link>
                    <p className="text-xs text-blue-400/50 mt-4">v1.2.0 (Modern Theme)</p>
                </div>
            </div>
        </div>
    )
}
