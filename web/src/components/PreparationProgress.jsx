import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../supabaseClient'

export default function PreparationProgress({ currentStep, batchId }) {
    const navigate = useNavigate()
    const [jobStatus, setJobStatus] = useState(null) // 'SUBMITTED', 'RUNNABLE', 'RUNNING', 'SUCCEEDED'
    const [jobsCompleted, setJobsCompleted] = useState(0)
    const [jobsTotal, setJobsTotal] = useState(0)

    const steps = [
        {
            id: 1,
            name: 'Protein Prepared',
            description: 'Parsing structure, identifying chains',
            icon: '🧬'
        },
        {
            id: 2,
            name: 'Water Removal',
            description: 'Stripping HOH/Water molecules',
            icon: '💧'
        },
        {
            id: 3,
            name: 'Ligand Prepared',
            description: '3D coordinates, charges',
            icon: '💊'
        },
        {
            id: 4,
            name: 'Config Generated',
            description: 'Grid parameters set',
            icon: '⚙️'
        },
        {
            id: 5,
            name: 'Grid File Ready',
            description: 'Binding site configured',
            icon: '📐'
        }
    ]

    // Poll job status after preparation is complete
    useEffect(() => {
        if (!batchId || currentStep < 5) return

        const pollStatus = async () => {
            try {
                const { data: { session } } = await supabase.auth.getSession()
                if (!session) return

                const response = await fetch(`${import.meta.env.VITE_API_URL || 'https://api.biodockify.com'}/jobs/batch/${batchId}`, {
                    headers: { 'Authorization': `Bearer ${session.access_token}` }
                })

                if (response.ok) {
                    const data = await response.json()

                    // Determine overall status
                    if (data.jobs && data.jobs.length > 0) {
                        const statuses = data.jobs.map(j => j.status)
                        const completed = statuses.filter(s => s === 'SUCCEEDED').length
                        const running = statuses.filter(s => s === 'RUNNING').length
                        const submitted = statuses.filter(s => s === 'SUBMITTED').length

                        setJobsCompleted(completed)
                        setJobsTotal(data.jobs.length)

                        if (completed === data.jobs.length) {
                            setJobStatus('SUCCEEDED')
                        } else if (running > 0) {
                            setJobStatus('RUNNING')
                        } else if (submitted > 0) {
                            setJobStatus('RUNNABLE') // Show as RUNNABLE when submitted
                        } else {
                            setJobStatus('SUBMITTED')
                        }
                    }
                }
            } catch (err) {
                console.error('Failed to poll job status:', err)
            }
        }

        pollStatus() // Initial poll
        const interval = setInterval(pollStatus, 5000) // Poll every 5 seconds

        return () => clearInterval(interval)
    }, [batchId, currentStep])

    return (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                <svg className="w-5 h-5 text-primary-600 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Preparing Files for Docking
            </h3>

            {/* Progress Bar */}
            <div className="mb-8">
                <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium text-slate-700">
                        Processing Step {currentStep} / 5
                    </span>
                    <span className="text-sm font-medium text-primary-600 animate-pulse">
                        {currentStep === 1 && "Analyzing PDB..."}
                        {currentStep === 2 && "Stripping Waters..."}
                        {currentStep === 3 && "Optimizing Ligands..."}
                        {currentStep === 4 && "Generating Config..."}
                        {currentStep === 5 && "Finalizing Batch..."}
                    </span>
                </div>
                <div className="flex gap-2">
                    {[1, 2, 3, 4, 5].map((step) => (
                        <div
                            key={step}
                            className={`h-2 flex-1 rounded-full transition-all duration-500 ${currentStep >= step
                                ? 'bg-gradient-to-r from-primary-500 to-primary-600 shadow-sm'
                                : 'bg-slate-200'
                                } ${currentStep === step ? 'animate-pulse' : ''}`}
                        ></div>
                    ))}
                </div>
            </div>

            {/* Steps List */}
            <div className="space-y-3">
                {steps.map((step) => {
                    const isCompleted = currentStep > step.id
                    const isActive = currentStep === step.id
                    const isPending = currentStep < step.id

                    return (
                        <div
                            key={step.id}
                            className={`flex items-start gap-4 p-4 rounded-lg border-2 transition-all ${isActive
                                ? 'border-primary-500 bg-primary-50'
                                : isCompleted
                                    ? 'border-green-500 bg-green-50'
                                    : 'border-slate-200 bg-slate-50 opacity-60'
                                }`}
                        >
                            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg ${isActive
                                ? 'bg-primary-500 text-white animate-pulse'
                                : isCompleted
                                    ? 'bg-green-500 text-white'
                                    : 'bg-slate-300 text-slate-600'
                                }`}>
                                {isCompleted ? (
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path>
                                    </svg>
                                ) : isActive ? (
                                    <svg className="w-6 h-6 animate-spin" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                ) : (
                                    step.icon
                                )}
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center gap-2">
                                    <div className={`font-bold ${isActive ? 'text-primary-700' : isCompleted ? 'text-green-700' : 'text-slate-600'
                                        }`}>
                                        {step.name}
                                    </div>
                                    {isActive && (
                                        <span className="px-2 py-0.5 bg-primary-100 text-primary-700 text-xs font-semibold rounded">
                                            In Progress
                                        </span>
                                    )}
                                    {isCompleted && (
                                        <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-semibold rounded">
                                            Complete
                                        </span>
                                    )}
                                </div>
                                <div className={`text-sm ${isActive ? 'text-primary-600' : isCompleted ? 'text-green-600' : 'text-slate-500'
                                    }`}>
                                    {step.description}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* Current Action */}
            {currentStep <= 5 && !jobStatus && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <div className="text-sm text-blue-900">
                            <strong>What's happening:</strong> {
                                currentStep === 1 ? "Parsing protein structure, identifying chains, and validating format." :
                                    currentStep === 2 ? "Detected water molecules (HOH). Stripping them to clear binding site." :
                                        currentStep === 3 ? "Generating 3D coordinates, energy minimization, and converting ligand." :
                                            currentStep === 4 ? "Creating AutoDock Vina configuration file with your grid box parameters." :
                                                "Finalizing setup and launching docker container."
                            }
                        </div>
                    </div>
                </div>
            )}

            {/* Job Status Bar (appears after step 5 is complete) */}
            {currentStep >= 5 && batchId && (
                <div className="mt-6 space-y-4">
                    <div className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
                        <h4 className="font-bold text-indigo-900 mb-3 flex items-center gap-2">
                            <span className="text-lg">🚀</span>
                            Job Execution Status
                        </h4>

                        <div className="space-y-3">
                            {/* Status Timeline */}
                            <div className="flex items-center gap-3">
                                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${jobStatus === 'SUBMITTED' || jobStatus === 'RUNNABLE' || jobStatus === 'RUNNING' || jobStatus === 'SUCCEEDED' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                                    <span className="text-xs font-bold">✓ SUBMITTED</span>
                                </div>
                                <div className="h-1 w-8 bg-gray-300 rounded"></div>
                                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${jobStatus === 'RUNNABLE' || jobStatus === 'RUNNING' || jobStatus === 'SUCCEEDED' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>
                                    <span className="text-xs font-bold">{jobStatus === 'RUNNABLE' ? '⏳ ' : jobStatus === 'RUNNING' || jobStatus === 'SUCCEEDED' ? '✓ ' : ''}RUNNABLE</span>
                                </div>
                                <div className="h-1 w-8 bg-gray-300 rounded"></div>
                                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${jobStatus === 'RUNNING' || jobStatus === 'SUCCEEDED' ? 'bg-purple-100 text-purple-700 animate-pulse' : 'bg-gray-100 text-gray-500'}`}>
                                    <span className="text-xs font-bold">{jobStatus === 'RUNNING' ? '🔄 ' : jobStatus === 'SUCCEEDED' ? '✓ ' : ''}RUNNING</span>
                                </div>
                            </div>

                            {/* Progress Info */}
                            {jobsTotal > 0 && (
                                <div className="text-sm text-indigo-700">
                                    <strong>Progress:</strong> {jobsCompleted} / {jobsTotal} ligands completed
                                </div>
                            )}
                        </div>
                    </div>

                    {/* View Live Result Button */}
                    {jobStatus && (
                        <button
                            onClick={() => navigate(`/dock/batch/${batchId}`)}
                            className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold rounded-xl shadow-lg transition-all hover:scale-105"
                        >
                            <span className="text-xl">📊</span>
                            <span>View Live Results</span>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                            </svg>
                        </button>
                    )}
                </div>
            )}
        </div>
    )
}
