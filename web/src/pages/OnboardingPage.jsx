import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FlaskConical, Database, Settings, Play, BarChart2, ArrowRight, CheckCircle2 } from 'lucide-react';

const OnboardingSteps = [
    {
        id: 1,
        title: '🧪 Upload Your Compounds',
        description: 'Paste SMILES or upload files (SDF, PDBQT). We support batch processing of up to 10,000 compounds at once.',
        example: 'CC(=O)Nc1ccccc1 (Aspirin)',
        icon: <FlaskConical className="w-12 h-12 text-blue-500" />,
        nextRoute: '/batch-docking'
    },
    {
        id: 2,
        title: '🎯 Select a Target',
        description: 'Choose a protein target from your project library or upload a new PDB structure (e.g., purified from PDB or AlphaFold).',
        example: 'GSK-3β (Alzheimer’s target)',
        icon: <Database className="w-12 h-12 text-purple-500" />,
        nextRoute: '/configure'
    },
    {
        id: 3,
        title: '⚙️ Configure Settings',
        description: 'Select your docking engine (Vina or Gnina) and enable advanced modules like MD Simulation or ADMET profiling.',
        example: 'Include MD: Yes, ADMET: Yes',
        icon: <Settings className="w-12 h-12 text-slate-500" />,
        nextRoute: '/batch'
    },
    {
        id: 4,
        title: '🚀 Run Simulation',
        description: 'Launch the high-performance computing pipeline. We handle the orchestration on AWS Batch.',
        example: 'Running... (Estimated: 2-3 hours)',
        icon: <Play className="w-12 h-12 text-emerald-500" />,
        nextRoute: '/batch'
    },
    {
        id: 5,
        title: '📊 Analyze & Export',
        description: 'Filter results by Binding Affinity, view 3D poses, check Toxicity risks, and export a PDF report.',
        example: 'Top Hit: -9.2 kcal/mol',
        icon: <BarChart2 className="w-12 h-12 text-orange-500" />,
        nextRoute: '/results'
    }
];

export default function OnboardingPage() {
    const [stepIndex, setStepIndex] = useState(0);
    const navigate = useNavigate();

    const step = OnboardingSteps[stepIndex];
    const progress = ((stepIndex + 1) / OnboardingSteps.length) * 100;

    const handleNext = () => {
        if (stepIndex === OnboardingSteps.length - 1) {
            finishOnboarding();
        } else {
            setStepIndex(stepIndex + 1);
        }
    };

    const finishOnboarding = () => {
        localStorage.setItem('biodockify_visited', 'true');
        navigate('/dock/new?source=onboarding');
    };

    const handleSkip = () => finishOnboarding();

    return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
            <div className="max-w-2xl w-full bg-white rounded-2xl shadow-xl overflow-hidden">

                {/* Progress Bar */}
                <div className="bg-slate-100 h-2">
                    <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-full transition-all duration-500 ease-out"
                        style={{ width: `${progress}%` }}
                    />
                </div>

                <div className="p-8 md:p-12">
                    <div className="flex justify-between items-center text-sm font-medium text-slate-400 mb-8">
                        <span>Step {stepIndex + 1} of {OnboardingSteps.length}</span>
                        <button onClick={handleSkip} className="hover:text-slate-600 transition-colors">Skip Tutorial</button>
                    </div>

                    <div className="flex flex-col items-center text-center space-y-6 mb-12">
                        <div className="w-24 h-24 bg-slate-50 rounded-full flex items-center justify-center mb-4 shadow-inner ring-8 ring-slate-50">
                            {step.icon}
                        </div>

                        <h1 className="text-3xl font-bold text-slate-900 tracking-tight">{step.title}</h1>

                        <p className="text-lg text-slate-600 max-w-lg leading-relaxed">
                            {step.description}
                        </p>

                        <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm text-slate-500 inline-flex items-center gap-2">
                            <span className="font-semibold text-slate-700">Example:</span> {step.example}
                        </div>
                    </div>

                    <div className="flex items-center justify-end">
                        <button
                            onClick={handleNext}
                            className="flex items-center gap-2 px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold transition-all hover:scale-105 shadow-lg shadow-indigo-200"
                        >
                            {stepIndex === OnboardingSteps.length - 1 ? 'Get Started' : 'Next Step'}
                            {stepIndex === OnboardingSteps.length - 1 ? <CheckCircle2 className="w-5 h-5" /> : <ArrowRight className="w-5 h-5" />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
