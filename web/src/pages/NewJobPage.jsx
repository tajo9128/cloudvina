import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import VerificationModal from '../components/VerificationModal'
import { API_URL } from '../config'

export default function NewJobPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFile, setLigandFile] = useState(null)
    const [error, setError] = useState(null)

    // Verification State
    const [isVerified, setIsVerified] = useState(false)
    const [checkingVerification, setCheckingVerification] = useState(true)
    const [isVerificationModalOpen, setIsVerificationModalOpen] = useState(false)
    const [userPhone, setUserPhone] = useState('')

    // Job Progress State
    const [submittedJob, setSubmittedJob] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const ESTIMATED_DURATION = 300 // 5 minutes

    useEffect(() => {
        checkVerification()
    }, [])

    const checkVerification = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const response = await fetch(`${API_URL}/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (response.ok) {
                const userData = await response.json()
                setIsVerified(userData.is_verified)
                setUserPhone(userData.phone)
            }
        } catch (err) {
            console.error('Error checking verification:', err)
        } finally {
            setCheckingVerification(false)
        }
    }

    useEffect(() => {
        let timer
        if (submittedJob && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status)) {
            const startTime = submittedJob.created_at ? new Date(submittedJob.created_at).getTime() : Date.now()
            timer = setInterval(() => {
                const now = Date.now()
                setElapsedTime(Math.floor((now - startTime) / 1000))
            }, 1000)
        }
        return () => clearInterval(timer)
    }, [submittedJob])

    // Poll for job status
    useEffect(() => {
        let poller
        if (submittedJob && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status)) {
            poller = setInterval(async () => {
                try {
                    const { data: { session } } = await supabase.auth.getSession()
                    if (!session) return

                    const res = await fetch(`${API_URL}/jobs/${submittedJob.job_id}`, {
                        headers: {
                            'Authorization': `Bearer ${session.access_token}`
                        }
                    })

                    if (res.ok) {
                        const updatedJob = await res.json()
                        setSubmittedJob(prev => ({
                            ...prev,
                            status: updatedJob.status
                        }))
                    }
                } catch (err) {
                    console.error('Polling error:', err)
                }
            }, 5000)
        }
        return () => clearInterval(poller)
    }, [submittedJob])

    const getProgressColor = (percentage) => {
        if (percentage < 30) return 'bg-blue-500'
        if (percentage < 70) return 'bg-purple-500'
        return 'bg-green-500'
    }

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!isVerified) {
            setIsVerificationModalOpen(true)
            return
        }

        if (!receptorFile || !ligandFile) {
            setError('Please select both receptor and ligand files')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // 1. Create Job
            const createRes = await fetch(`${API_URL}/jobs/submit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filename: ligandFile.name
                })
            })

            if (!createRes.ok) {
                const err = await createRes.json()
                if (createRes.status === 403 && err.detail?.reason === 'phone_not_verified') {
                    setIsVerified(false)
                    setIsVerificationModalOpen(true)
                    throw new Error('Phone verification required')
                }
                throw new Error(err.detail?.message || err.detail || 'Failed to create job')
            }

            const { job_id, upload_urls } = await createRes.json()

            // 2. Upload Files
            await uploadFile(upload_urls.receptor, receptorFile)
            await uploadFile(upload_urls.ligand, ligandFile)

            // 3. Start Job
            const startRes = await fetch(`${API_URL}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!startRes.ok) throw new Error('Failed to start job')

            // Instead of navigating, set state to show progress
            setSubmittedJob({
                job_id,
                status: 'SUBMITTED',
                created_at: new Date().toISOString()
            })

        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const uploadFile = async (url, file) => {
        const res = await fetch(url, {
            method: 'PUT',
            body: file,
            headers: {
                'Content-Type': file.type || 'application/octet-stream'
            }
        })
        if (!res.ok) throw new Error(`Failed to upload ${file.name}`)
    }

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <VerificationModal
                isOpen={isVerificationModalOpen}
                onClose={() => setIsVerificationModalOpen(false)}
                userPhone={userPhone}
                onVerified={() => {
                    setIsVerified(true)
                    setIsVerificationModalOpen(false)
                }}
            />

            <main className="container mx-auto px-4 py-12">
                <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
                    <h1 className="text-2xl font-bold text-gray-900 mb-6">Start New Docking Job</h1>

                    {!checkingVerification && !isVerified && (
                        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
                            <div className="flex items-center">
                                <div className="flex-shrink-0">
                                    ⚠️
                                </div>
                                <div className="ml-3">
                                    <p className="text-sm text-yellow-700">
                                        Phone verification is required to submit jobs.
                                    </p>
                                    <button
                                        onClick={() => setIsVerificationModalOpen(true)}
                                        className="mt-2 text-sm font-bold text-yellow-700 hover:text-yellow-800 underline"
                                    >
                                        Verify Now
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Receptor Upload */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Receptor (PDB)
                            </label>
                            <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors relative ${submittedJob ? 'bg-gray-50 border-gray-200' : 'border-gray-300 hover:border-purple-500 cursor-pointer'}`}>
                                <input
                                    type="file"
                                    accept=".pdb"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                                />
                                <div className="space-y-1">
                                    <div className="text-3xl">🧬</div>
                                    {receptorFile ? (
                                        <div className="text-purple-600 font-medium">{receptorFile.name}</div>
                                    ) : (
                                        <div className="text-gray-500">Upload Receptor (.pdb)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Ligand Upload */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Ligand (PDBQT, SDF, MOL2)
                            </label>
                            <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors relative ${submittedJob ? 'bg-gray-50 border-gray-200' : 'border-gray-300 hover:border-purple-500 cursor-pointer'}`}>
                                <input
                                    type="file"
                                    accept=".pdbqt,.sdf,.mol2"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setLigandFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                                />
                                <div className="space-y-1">
                                    <div className="text-3xl">💊</div>
                                    {ligandFile ? (
                                        <div className="text-purple-600 font-medium">{ligandFile.name}</div>
                                    ) : (
                                        <div className="text-gray-500">Upload Ligand (.pdbqt, .sdf)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {error && (
                            <div className="bg-red-50 text-red-600 p-4 rounded-lg text-sm">
                                {error}
                            </div>
                        )}

                        {!submittedJob ? (
                            <button
                                type="submit"
                                disabled={loading}
                                className={`w-full py-3 rounded-lg font-bold text-white transition ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700 shadow-md'
                                    }`}
                            >
                                {loading ? 'Submitting...' : 'Launch Docking Job'}
                            </button>
                        ) : (
                            <div className="space-y-4 animate-in fade-in duration-500">
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="font-medium text-gray-700">Status: {submittedJob.status}</span>
                                        <span className="text-sm text-gray-500">Est. Remaining: {formatTime(remainingSeconds)}</span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="w-full bg-gray-200 rounded-full h-3 mb-2 overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-500 ${getProgressColor(progressPercentage)}`}
                                            style={{ width: `${progressPercentage}%` }}
                                        >
                                            {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status) && (
                                                <div className="w-full h-full animate-pulse bg-white/20"></div>
                                            )}
                                        </div>
                                    </div>

                                    <p className="text-xs text-gray-500 text-center">
                                        Job ID: {submittedJob.job_id}
                                    </p>
                                </div>

                                {submittedJob.status === 'SUCCEEDED' && (
                                    <button
                                        type="button"
                                        onClick={() => navigate(`/dock/${submittedJob.job_id}`)}
                                        className="w-full py-3 rounded-lg font-bold text-white bg-green-600 hover:bg-green-700 shadow-md transition animate-bounce"
                                    >
                                        View Results 🎉
                                    </button>
                                )}

                                {submittedJob.status === 'FAILED' && (
                                    <div className="text-center text-red-600 font-medium">
                                        Job Failed. Please try again.
                                    </div>
                                )}
                            </div>
                        )}
                    </form>
                </div>
            </main>
        </div>
    )
}
