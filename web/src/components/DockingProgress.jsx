import React from 'react'

export default function DockingProgress({ status, jobId, steps, currentStep, detail }) {
    // Calculate progress for each track based on current step
    // Steps: 0=Prep, 1=Vina, 2=Gnina, 3=Consensus, 4=Finalizing

    const getProgress = (targetStep) => {
        if (currentStep > targetStep) return 100
        if (currentStep === targetStep) return 60 // Active
        return 0
    }

    const isActive = (targetStep) => currentStep === targetStep

    return (
        <div className="w-full max-w-3xl mx-auto bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden transform transition-all duration-500">
            {/* 1. Status Bar */}
            <div className="bg-slate-900 text-white px-6 py-3 flex justify-between items-center text-sm font-mono border-b border-primary-500/30">
                <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                    <span>JOB ID: <span className="text-secondary-400">{jobId || 'Initializing...'}</span></span>
                </div>
                <div className="text-slate-400">{status || 'Processing...'}</div>
            </div>

            <div className="p-8 space-y-6">
                {/* 2. Preparation Track */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase text-slate-500">
                        <span className={isActive(0) || isActive(1) ? "text-primary-600" : ""}>Step 1: System Preparation</span>
                        <span>{getProgress(1)}%</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-blue-500 transition-all duration-1000 ease-out"
                            style={{ width: `${getProgress(1)}%` }}
                        ></div>
                    </div>
                    <p className="text-xs text-slate-400">
                        {currentStep <= 1 ? "Converting PDB to PDBQT, cleaning waters, adding polar hydrogens..." : "Preparation Complete"}
                    </p>
                </div>

                {/* 3. Parallel Engines Track */}
                <div className="grid grid-cols-2 gap-8 relative">
                    {/* Connector */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-slate-100 border border-slate-300 flex items-center justify-center z-10">
                        <span className="text-xs font-bold text-slate-400">+</span>
                    </div>

                    {/* Vina */}
                    <div className={`p-4 rounded-lg border ${isActive(2) ? 'border-primary-500 bg-primary-50/50 ring-1 ring-primary-200' : 'border-slate-200 bg-slate-50'}`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">⚡</span>
                            <h4 className="font-bold text-slate-700">AutoDock Vina</h4>
                        </div>
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden mb-2">
                            <div
                                className="h-full bg-orange-500"
                                style={{ width: `${getProgress(2)}%` }}
                            ></div>
                        </div>
                        <p className="text-xs text-slate-500">
                            {isActive(2) ? "Running Genetic Algorithm..." : getProgress(2) === 100 ? "Optimization Complete" : "Waiting..."}
                        </p>
                    </div>

                    {/* Gnina */}
                    <div className={`p-4 rounded-lg border ${isActive(2) ? 'border-secondary-500 bg-secondary-50/50 ring-1 ring-secondary-200' : 'border-slate-200 bg-slate-50'}`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">🧠</span>
                            <h4 className="font-bold text-slate-700">Gnina (AI)</h4>
                        </div>
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden mb-2">
                            <div
                                className="h-full bg-purple-500"
                                style={{ width: `${getProgress(2)}%` }}
                            ></div>
                        </div>
                        <p className="text-xs text-slate-500">
                            {isActive(2) ? "Running CNN Scoring..." : getProgress(2) === 100 ? "Inference Complete" : "Waiting..."}
                        </p>
                    </div>
                </div>

                {/* 4. Consensus Track */}
                <div className="space-y-2 pt-2">
                    <div className="flex justify-between text-xs font-bold uppercase text-slate-500">
                        <span className={isActive(3) ? "text-green-600" : ""}>Step 3: Consensus Aggregation</span>
                        <span>{getProgress(3)}%</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-green-500 transition-all duration-500"
                            style={{ width: `${getProgress(3)}%` }}
                        ></div>
                    </div>
                    <p className="text-xs text-slate-400">
                        {currentStep >= 3 ? "Merging scores and ranking poses..." : "Waiting for engines..."}
                    </p>
                </div>

                {detail && (
                    <div className="text-center text-xs text-slate-400 pt-2 border-t border-slate-100">
                        Latest: {detail}
                    </div>
                )}
            </div>
        </div>
    )
}
