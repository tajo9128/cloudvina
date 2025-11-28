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
                        setSuccess('✅ Account created! Please check your email to verify your account.')
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
            setError(error.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-600 rounded-2xl mb-4 shadow-lg">
                        <span className="text-3xl">🧬</span>
                    </div>
                    <h2 className="text-3xl font-extrabold text-gray-900">
                        {isSignUp ? 'Create your account' : 'Welcome back'}
                    </h2>
                    <p className="mt-2 text-sm text-gray-600">
                        {isSignUp ? 'Start your molecular docking journey' : 'Sign in to continue to CloudVina'}
                    </p>
                </div>

                {/* Form Card */}
                <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
                    <form className="space-y-5" onSubmit={handleAuth}>
                        {/* Email */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Email Address *
                            </label>
                            <input
                                type="email"
                                required
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                placeholder="you@university.edu"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                        </div>

                        {/* Password */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Password *
                            </label>
                            <input
                                type="password"
                                required
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                            {isSignUp && (
                                <p className="mt-1 text-xs text-gray-500">Minimum 6 characters</p>
                            )}
                        </div>

                        {/* Signup-only fields */}
                        {isSignUp && (
                            <>
                                {/* Info Box */}
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                    <div className="flex">
                                        <div className="flex-shrink-0">
                                            <svg className="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                            </svg>
                                        </div>
                                        <div className="ml-3">
                                            <p className="text-xs text-blue-700">
                                                <strong>Email verification required.</strong> Free users get 3 docking jobs per day. Upgrade anytime for unlimited access.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}
                    </>
                        )}

                    {/* Error/Success Messages */}
                    {error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                            <p className="text-sm text-red-600">❌ {error}</p>
                        </div>
                    )}

                    {success && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                            <p className="text-sm text-green-600">{success}</p>
                        </div>
                    )}

                    {/* Submit Button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
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
                        className="text-sm text-purple-600 hover:text-purple-700 font-medium"
                    >
                        {isSignUp ? '← Already have an account? Sign in' : "Don't have an account? Sign up →"}
                    </button>
                </div>
            </div>

            {/* Footer */}
            <div className="mt-6 text-center">
                <Link to="/" className="text-sm text-gray-600 hover:text-gray-900">
                    ← Back to Homepage
                </Link>
                <p className="text-xs text-gray-400 mt-4">v1.1.2 (Crash Fixes Applied)</p>
            </div>
        </div>
        </div >
    )
}
