import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import PreparationProgress from '../components/PreparationProgress'

export default function NewJobPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFile, setLigandFile] = useState(null)
    const [error, setError] = useState(null)
    const [preparationStep, setPreparationStep] = useState(0) // 0=not started, 1-4=prep steps

    // Job Progress State
    const [submittedJob, setSubmittedJob] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const ESTIMATED_DURATION = 300 // 5 minutes

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
        if (percentage < 30) return 'bg-primary-500'
        if (percentage < 70) return 'bg-secondary-500'
        return 'bg-green-500'
    }

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
    }

    // Helper: Fetch with retry logic for cold starts (up to 90s wait)
    const fetchWithRetry = async (url, options, retries = 15, backoff = 2000) => {
        try {
            const res = await fetch(url, options)
            if (!res.ok) {
                // If 503 or 504 (Service Unavailable/Gateway Timeout), retry
                if ([503, 504].includes(res.status) && retries > 0) {
                    console.log(`Server starting... Retrying ${url} in ${backoff / 1000}s... Attempts left: ${retries}`)
                    await new Promise(r => setTimeout(r, backoff))
                    // Cap backoff at 5 seconds to poll frequently
                    const nextBackoff = Math.min(backoff * 1.5, 5000)
                    return fetchWithRetry(url, options, retries - 1, nextBackoff)
                }
                return res
            }
            return res
        } catch (err) {
            // Retry on network errors (like Failed to fetch)
            if (retries > 0) {
                console.log(`Network error (Server likely sleeping)... Retrying in ${backoff / 1000}s... Attempts left: ${retries}`)
                await new Promise(r => setTimeout(r, backoff))
                const nextBackoff = Math.min(backoff * 1.5, 5000)
                return fetchWithRetry(url, options, retries - 1, nextBackoff)
            }
            throw err
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!receptorFile || !ligandFile) {
            setError('Please select both receptor and ligand files')
            return
        }

        setLoading(true)
        setError('⏰ Connecting to server... (This may take up to 60s if server is sleeping)')

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // 1. Create Job with Retry
            console.log('Step 1: Creating job with backend...')
            const createRes = await fetchWithRetry(`${API_URL}/jobs/submit`, {
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
            console.log('Step 1 response:', createRes.status, createRes.ok)

            if (!createRes.ok) {
                const err = await createRes.json()
                const errorMessage = typeof err.detail === 'object'
                    ? JSON.stringify(err.detail)
                    : (err.detail?.message || err.detail || 'Failed to create job')
                throw new Error(errorMessage)
            }

            const { job_id, upload_urls } = await createRes.json()
            console.log('Step 1 complete. Job ID:', job_id)

            // 2. Upload Files (No retry needed usually, S3 is reliable)
            console.log('Step 2: Uploading files to S3...')
            setError('📤 Uploading files...')
            await uploadFile(upload_urls.receptor, receptorFile)
            console.log('Receptor uploaded')
            await uploadFile(upload_urls.ligand, ligandFile)
            console.log('Ligand uploaded')

            // 2.5 Show preparation progress
            setError(null)
            setPreparationStep(1) // Protein preparation
            await new Promise(r => setTimeout(r, 1500))

            setPreparationStep(2) // Ligand preparation
            await new Promise(r => setTimeout(r, 1500))

            setPreparationStep(3) // Config generation
            await new Promise(r => setTimeout(r, 1000))

            setPreparationStep(4) // Grid file ready
            await new Promise(r => setTimeout(r, 1000))

            // 3. Start Job with Retry
            console.log('Step 3: Starting job...')
            setError('🚀 Starting simulation...')
            const startRes = await fetchWithRetry(`${API_URL}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })
            console.log('Step 3 response:', startRes.status, startRes.ok)

            if (!startRes.ok) throw new Error('Failed to start job')

            console.log('Job submitted successfully!')
            // Instead of navigating, set state to show progress
            setSubmittedJob({
                job_id,
                status: 'SUBMITTED',
                created_at: new Date().toISOString()
            })

        } catch (err) {
            console.error('Submission error:', err)
            // Check if it's an email verification error
            if (err.message.includes('verify your email')) {
                setError(
                    <div className="flex flex-col items-start gap-2">
                        <span>{err.message}</span>
                        <button
                            onClick={async () => {
                                setLoading(true)
                                const { data, error } = await supabase.auth.refreshSession()
                                if (!error && data.session?.user?.email_confirmed_at) {
                                    setError(null)
                                    alert('Email verified successfully! You can now submit jobs.')
                                } else {
                                    alert('Email still not verified. Please check your inbox or try logging out and back in.')
                                }
                                setLoading(false)
                            }}
                            className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded border border-red-200 hover:bg-red-200 transition"
                        >
                            🔄 I have verified my email
                        </button>
                    </div>
                )
            } else {
                setError(`Error: ${err.message}. Please try again.`)
            }
        } finally {
            setLoading(false)
        }
    }

    const uploadFile = async (url, file) => {
        // Extract content-type from presigned URL query params
        // S3 presigned URLs include the ContentType as a query parameter
        // We MUST use the exact same Content-Type in the PUT request or we get 403
        const urlObj = new URL(url)
        const contentType = urlObj.searchParams.get('content-type') ||
            urlObj.searchParams.get('ContentType') ||
            file.type ||
            'application/octet-stream'

        const res = await fetch(url, {
            method: 'PUT',
            body: file,
            headers: {
                'Content-Type': contentType
            }
        })
        if (!res.ok) throw new Error(`Failed to upload ${file.name}`)
    }

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-2xl mx-auto bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
                    <div className="mb-8 text-center">
                        <div className="inline-block p-3 rounded-full bg-primary-50 text-primary-600 mb-4">
                            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                        </div>
                        <h1 className="text-3xl font-bold text-slate-900 mb-2">New Docking Simulation</h1>
                        <p className="text-slate-500">Upload your receptor and ligand files to begin.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-8">
                        {/* Receptor Upload */}
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">
                                Receptor File <span className="text-slate-400 font-normal ml-1">(PDB, PDBQT, MOL2, SDF, CIF, XML)</span>
                            </label>
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all relative group ${submittedJob ? 'bg-slate-50 border-slate-200' : 'border-slate-300 hover:border-primary-500 hover:bg-primary-50/50 cursor-pointer bg-slate-50'}`}>
                                <input
                                    type="file"
                                    accept=".pdb,.pdbqt,.mol2,.sdf,.mol,.cif,.mmcif,.pqr,.xml,.pdbml,.gz"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-20"
                                />
                                <div className="space-y-3 relative z-10">
                                    <div className="w-12 h-12 bg-white rounded-full shadow-sm border border-slate-100 flex items-center justify-center mx-auto text-2xl group-hover:scale-110 transition-transform">🧬</div>
                                    {receptorFile ? (
                                        <div className="text-primary-600 font-bold text-lg flex items-center justify-center gap-2">
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                            {receptorFile.name}
                                        </div>
                                    ) : (
                                        <div>
                                            <div className="text-slate-900 font-medium">Click to upload receptor</div>
                                            <div className="text-slate-400 text-sm mt-1">Supports .gz compression</div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Ligand Upload */}
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">
                                Ligand File <span className="text-slate-400 font-normal ml-1">(PDBQT, SDF, MOL2, SMILES)</span>
                            </label>
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all relative group ${submittedJob ? 'bg-slate-50 border-slate-200' : 'border-slate-300 hover:border-secondary-500 hover:bg-secondary-50/50 cursor-pointer bg-slate-50'}`}>
                                <input
                                    type="file"
                                    accept=".pdbqt,.sdf,.mol2,.mol,.smi,.smiles,.gz"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setLigandFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-20"
                                />
                                <div className="space-y-3 relative z-10">
                                    <div className="w-12 h-12 bg-white rounded-full shadow-sm border border-slate-100 flex items-center justify-center mx-auto text-2xl group-hover:scale-110 transition-transform">💊</div>
                                    {ligandFile ? (
                                        <div className="text-secondary-600 font-bold text-lg flex items-center justify-center gap-2">
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                            {ligandFile.name}
                                        </div>
                                    ) : (
                                        <div>
                                            <div className="text-slate-900 font-medium">Click to upload ligand</div>
                                            <div className="text-slate-400 text-sm mt-1">Supports .gz compression</div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Preparation Progress */}
                        {preparationStep > 0 && !submittedJob && (
                            <PreparationProgress currentStep={preparationStep} />
                        )}

                        {error && (
                            <div className="bg-red-50 border border-red-100 text-red-600 p-4 rounded-xl text-sm flex items-start gap-3">
                                <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                <div>{error}</div>
                            </div>
                        )}

                        {!submittedJob ? (
                            <button
                                type="submit"
                                disabled={loading}
                                className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg shadow-primary-600/20 ${loading
                                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                    : 'bg-primary-600 text-white hover:bg-primary-700 hover:-translate-y-1'}`}
                            >
                                {loading ? (
                                    <span className="flex items-center justify-center gap-2">
                                        <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Initializing Simulation...
                                    </span>
                                ) : 'Launch Docking Simulation'}
                            </button>
                        ) : (
                            <div className="space-y-6 animate-in fade-in duration-500">
                                <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                                    <div className="flex justify-between items-center mb-4">
                                        <span className="font-bold text-slate-700 flex items-center gap-2">
                                            Status:
                                            <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${submittedJob.status === 'SUCCEEDED' ? 'bg-green-100 text-green-700' : 'bg-primary-100 text-primary-700'}`}>
                                                {submittedJob.status}
                                            </span>
                                        </span>
                                        <span className="text-sm text-slate-500 font-mono">Est. Remaining: {formatTime(remainingSeconds)}</span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="w-full bg-slate-200 rounded-full h-3 mb-4 overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-500 bg-gradient-to-r from-primary-500 to-secondary-500`}
                                            style={{ width: `${progressPercentage}%` }}
                                        >
                                            {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status) && (
                                                <div className="w-full h-full animate-pulse bg-white/30"></div>
                                            )}
                                        </div>
                                    </div>

                                    <p className="text-xs text-slate-400 text-center font-mono">
                                        Job ID: {submittedJob.job_id}
                                    </p>
                                </div>

                                {submittedJob.status === 'SUCCEEDED' && (
                                    <button
                                        type="button"
                                        onClick={() => navigate(`/dock/${submittedJob.job_id}`)}
                                        className="w-full py-4 rounded-xl font-bold text-white bg-green-600 hover:bg-green-700 shadow-lg shadow-green-600/20 transition animate-bounce flex items-center justify-center gap-2"
                                    >
                                        View Results 🎉
                                    </button>
                                )}

                                {submittedJob.status === 'FAILED' && (
                                    <div className="text-center text-red-600 font-bold bg-red-50 p-4 rounded-xl border border-red-100">
                                        Simulation Failed. Please try again.
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
