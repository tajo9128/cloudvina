import { useState } from 'react'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'

export default function VerificationModal({ isOpen, onClose, userPhone, onVerified }) {
    const [otp, setOtp] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [step, setStep] = useState('send') // 'send' or 'verify'
    const [devOtp, setDevOtp] = useState(null) // For displaying mock OTP in dev mode

    if (!isOpen) return null

    const handleSendOtp = async () => {
        setLoading(true)
        setError(null)
        try {
            const { data: { session } } = await supabase.auth.getSession()

            if (!session) {
                throw new Error('Please log in to verify your phone number.')
            }

            const response = await fetch(`${API_URL}/auth/send-otp`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({ phone: userPhone })
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to send OTP')
            }

            setStep('verify')
            if (data.dev_mode_otp) {
                setDevOtp(data.dev_mode_otp)
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const handleVerifyOtp = async () => {
        setLoading(true)
        setError(null)
        try {
            const { data: { session } } = await supabase.auth.getSession()

            if (!session) {
                throw new Error('Please log in to verify your phone number.')
            }

            const response = await fetch(`${API_URL}/auth/verify-otp`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({ phone: userPhone, otp })
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.detail || 'Verification failed')
            }

            onVerified()
            onClose()
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6 relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
                >
                    ✕
                </button>

                <div className="text-center mb-6">
                    <div className="text-4xl mb-2">📱</div>
                    <h2 className="text-2xl font-bold text-gray-900">Verify Phone Number</h2>
                    <p className="text-gray-600 mt-2">
                        To prevent abuse of free credits, we need to verify your phone number.
                    </p>
                </div>

                {error && (
                    <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4 text-sm">
                        {error}
                    </div>
                )}

                {step === 'send' ? (
                    <div className="space-y-4">
                        <div className="bg-gray-50 p-4 rounded-lg text-center">
                            <p className="text-sm text-gray-500 mb-1">Verifying number:</p>
                            <p className="font-mono font-bold text-lg">{userPhone || "No phone number found"}</p>
                        </div>
                        <button
                            onClick={handleSendOtp}
                            disabled={loading || !userPhone}
                            className="w-full bg-purple-600 text-white py-3 rounded-lg font-bold hover:bg-purple-700 transition disabled:opacity-50"
                        >
                            {loading ? 'Sending...' : 'Send OTP'}
                        </button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {devOtp && (
                            <div className="bg-green-50 border border-green-200 p-3 rounded-lg text-center mb-4">
                                <p className="text-xs text-green-800 font-bold uppercase mb-1">Dev Mode</p>
                                <p className="text-sm text-green-700">Your OTP is: <span className="font-mono font-bold text-lg">{devOtp}</span></p>
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Enter 6-digit OTP
                            </label>
                            <input
                                type="text"
                                maxLength="6"
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg text-center text-2xl tracking-widest focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                placeholder="000000"
                                value={otp}
                                onChange={(e) => setOtp(e.target.value.replace(/[^0-9]/g, ''))}
                            />
                        </div>
                        <button
                            onClick={handleVerifyOtp}
                            disabled={loading || otp.length !== 6}
                            className="w-full bg-purple-600 text-white py-3 rounded-lg font-bold hover:bg-purple-700 transition disabled:opacity-50"
                        >
                            {loading ? 'Verifying...' : 'Verify & Continue'}
                        </button>
                        <button
                            onClick={() => setStep('send')}
                            className="w-full text-gray-500 text-sm hover:text-gray-700"
                        >
                            Resend OTP
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
