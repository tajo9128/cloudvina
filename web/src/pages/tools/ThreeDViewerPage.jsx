import React, { useState } from 'react';
import MoleculeViewer from '../../components/MoleculeViewer';
import SEOHelmet from '../../components/SEOHelmet';

export default function ThreeDViewerPage() {
    const [structureData, setStructureData] = useState(null);
    const [fileName, setFileName] = useState('');

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setFileName(file.name);
            const reader = new FileReader();
            reader.onload = (event) => {
                setStructureData(event.target.result);
            };
            reader.readAsText(file);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 py-12 px-4">
            <SEOHelmet
                title="Free Online 3D Molecule Viewer | PDB & PDBQT"
                description="Visualize molecular structures (PDB, PDBQT) instantly in your browser. No installation required."
                keywords="3d molecule viewer, pdb viewer online, pdbqt viewer, molecular visualization"
                canonical="https://biodockify.com/3d-viewer"
            />
            <div className="max-w-6xl mx-auto h-[80vh] flex flex-col">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-900">Standalone 3D Viewer</h1>
                        <p className="text-slate-500">Upload PDB or PDBQT files to visualize instantly.</p>
                    </div>
                    <div className="relative">
                        <input
                            type="file"
                            accept=".pdb,.pdbqt"
                            onChange={handleFileUpload}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <button className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg transition-all">
                            📂 Upload Structure
                        </button>
                    </div>
                </div>

                <div className="flex-1 bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden relative">
                    {structureData ? (
                        <>
                            <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur px-3 py-1 rounded border border-slate-200 text-sm font-bold text-slate-700">
                                {fileName}
                            </div>
                            <MoleculeViewer
                                pdbqtData={fileName.endsWith('qt') ? structureData : null}
                                receptorData={!fileName.endsWith('qt') ? structureData : null}
                                width="100%"
                                height="100%"
                            />
                        </>
                    ) : (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 bg-slate-50/50">
                            <div className="text-6xl mb-4">🧊</div>
                            <p className="text-xl font-medium">No structure loaded</p>
                            <p className="text-sm mt-2">Upload a file to begin</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
