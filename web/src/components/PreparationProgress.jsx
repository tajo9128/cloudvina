export default function PreparationProgress({ currentStep }) {
    const steps = [
        {
            id: 1,
            name: 'Protein Prepared',
            description: 'Cleaning, adding hydrogens',
            icon: '🧬'
        },
        {
            id: 2,
            name: 'Ligand Prepared',
            description: '3D coordinates, charges',
            icon: '💊'
        },
        {
            id: 3,
            name: 'Config Generated',
            description: 'Grid parameters set',
            icon: '⚙️'
        },
        {
            id: 4,
            name: 'Grid File Ready',
            description: 'Binding site configured',
            icon: '📐'
        }
    ]

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
                        Step {currentStep} of 4
                    </span>
                    <span className="text-sm font-medium text-primary-600">
                        {Math.round((currentStep / 4) * 100)}%
                    </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                    <div
                        className="bg-primary-600 h-2 rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${(currentStep / 4) * 100}%` }}
                    ></div>
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
            {currentStep <= 4 && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-start gap-3">
                        <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <div className="text-sm text-blue-900">
                            <strong>What's happening:</strong> {
                                currentStep === 1 ? "Removing water molecules, adding polar hydrogens, and assigning charges to the protein structure." :
                                    currentStep === 2 ? "Generating 3D coordinates, energy minimization, and converting ligand to docking-ready format." :
                                        currentStep === 3 ? "Creating AutoDock Vina configuration file with your grid box parameters." :
                                            "Setting up the binding site grid and validating all files are ready for docking."
                            }
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
