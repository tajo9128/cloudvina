import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Database, BrainCircuit, Play } from 'lucide-react';
import { projectService } from '../services/projectService';
import { compoundService } from '../services/compoundService';
import { trainModel } from '../services/aiService';

// Modular Components
import CSVUploader from '../components/CompoundUpload/CSVUploader';
import CompoundPreview from '../components/CompoundUpload/CompoundPreview';
import ModelBuilder from '../components/QSARTraining/ModelBuilder';
import ModelStats from '../components/QSARTraining/ModelStats';
import AutoQSARPanel from '../components/QSARTraining/AutoQSARPanel';
import PredictionSandbox from '../components/Predictions/PredictionSandbox';

export default function QSARPage() {
    const { projectId } = useParams();
    const navigate = useNavigate();

    // State
    const [project, setProject] = useState(null);
    const [activeTab, setActiveTab] = useState('data'); // data | train | predict
    const [compounds, setCompounds] = useState([]);
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);

    // Training State
    const [trainStatus, setTrainStatus] = useState('idle');

    useEffect(() => {
        loadProjectData();
    }, [projectId]);

    const loadProjectData = async () => {
        try {
            const p = await projectService.getProjectDetails(projectId);
            setProject(p);

            const c = await compoundService.getCompounds(projectId);
            setCompounds(c);

            // TODO: Add getModels to projectService or aiService
            // const m = await projectService.getModels(projectId); 
            // setModels(m); 
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleCSVUpload = async (csvText) => {
        try {
            // Use compoundService for parsing AND uploading
            const newCompounds = await compoundService.uploadCompounds(projectId, csvText);
            // setCompounds(prev => [...prev, ...newCompounds]); // Append
            loadProjectData(); // Or just reload to be safe
            alert(`Uploaded ${newCompounds.length} compounds successfully!`);
        } catch (error) {
            console.error(error);
            alert(`Upload failed: ${error.message}`);
        }
    };

    const handleTrainModel = async (config) => {
        if (!compounds.length) return alert("Upload data first!");

        setTrainStatus('training');
        try {
            const smiles = compounds.map(c => c.smiles);
            const targets = compounds.map(c => parseFloat(c.properties[config.target] || 0));

            await trainModel(projectId, config.name, smiles, targets);
            setTrainStatus('success');
            loadProjectData(); // Refresh models list
        } catch (e) {
            console.error(e);
            setTrainStatus('error');
        }
    };

    if (loading) return <div className="flex h-screen items-center justify-center text-indigo-600">Loading Workspace...</div>;

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col">
            {/* Top Bar */}
            <header className="bg-white border-b border-slate-200 h-16 flex items-center px-6 sticky top-0 z-20">
                <button onClick={() => navigate('/dashboard')} className="mr-4 p-2 hover:bg-slate-100 rounded-lg text-slate-500">
                    <ArrowLeft size={20} />
                </button>
                <div>
                    <h1 className="text-lg font-bold text-slate-900">{project?.name}</h1>
                    <span className="text-xs text-slate-500 uppercase tracking-wider">QSAR Workspace</span>
                </div>
            </header>

            <div className="flex flex-1 overflow-hidden">
                {/* Sidebar Navigation */}
                <nav className="w-64 bg-white border-r border-slate-200 flex flex-col p-4 gap-2">
                    <NavItem
                        icon={<Database size={18} />}
                        label="Data Management"
                        active={activeTab === 'data'}
                        onClick={() => setActiveTab('data')}
                    />
                    <NavItem
                        icon={<BrainCircuit size={18} />}
                        label="Model Training"
                        active={activeTab === 'train'}
                        onClick={() => setActiveTab('train')}
                    />
                    <NavItem
                        icon={<Play size={18} />}
                        label="Predictions"
                        active={activeTab === 'predict'}
                        onClick={() => setActiveTab('predict')}
                    />
                </nav>

                {/* Main Canvas */}
                <main className="flex-1 overflow-auto p-8">

                    {/* DATA TAB */}
                    {activeTab === 'data' && (
                        <div className="max-w-4xl mx-auto">
                            <h2 className="text-2xl font-bold mb-6 text-slate-800">Dataset</h2>
                            <CSVUploader onUpload={handleCSVUpload} />
                            <CompoundPreview compounds={compounds} />
                        </div>
                    )}

                    {/* TRAIN TAB */}
                    {activeTab === 'train' && (
                        <div className="max-w-3xl mx-auto">
                            <div className="mb-6">
                                <h2 className="text-2xl font-bold text-slate-800">Train New Model</h2>
                                <p className="text-slate-500">Upload a dataset to automatically train a QSAR model.</p>
                            </div>
                            <AutoQSARPanel />
                        </div>
                    )}

                    {/* PREDICT TAB */}
                    {activeTab === 'predict' && (
                        <div className="max-w-4xl mx-auto">
                            <h2 className="text-2xl font-bold mb-6 text-slate-800">Models & Predictions</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <ModelStats models={models} />
                                <PredictionSandbox />
                            </div>
                        </div>
                    )}

                </main>
            </div>
        </div>
    );
}

function NavItem({ icon, label, active, onClick }) {
    return (
        <button
            onClick={onClick}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${active ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-50'}
            `}
        >
            {icon}
            {label}
        </button>
    );
}
