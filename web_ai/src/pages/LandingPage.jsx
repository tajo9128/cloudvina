import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
    const navigate = useNavigate();
    const [selectedPlan, setSelectedPlan] = useState(null);

    // Plan A State
    const [planADisease, setPlanADisease] = useState('');
    const [planATarget, setPlanATarget] = useState('');
    const [planACompounds, setPlanACompounds] = useState('');

    // Plan B State
    const [planBPlant, setPlanBPlant] = useState('');
    const [planBDisease, setPlanBDisease] = useState('');

    const handlePlanSelect = (plan) => {
        setSelectedPlan(plan);
    };

    const startPlanA = () => {
        if (!planADisease || !planATarget) {
            alert('Please select a disease and target first.');
            return;
        }
        // Navigate to dashboard with context
        // In a real app, you'd pass this state via context or query params
        console.log("Starting Plan A:", { planADisease, planATarget, planACompounds });
        navigate('/dashboard');
    };

    const startPlanB = () => {
        if (!planBPlant || !planBDisease) {
            alert('Please select a plant and disease first.');
            return;
        }
        console.log("Starting Plan B:", { planBPlant, planBDisease });
        navigate('/dashboard');
    };

    const goToFormulation = () => {
        navigate('/formulation');
    };

    const goToMain = () => {
        window.location.href = 'https://biodockify.com'; // External or different subdomain link
    };

    return (
        <div className="bg-[#f9faf8] min-h-screen text-[#134252] font-sans">
            {/* Breadcrumb - mocked as part of header in layout, but added here if needed matching HTML */}
            <div className="max-w-[1400px] mx-auto px-8 py-4 text-sm text-[#5c6b72]">
                Home &gt; Plan Selection
            </div>

            <div className="max-w-[1400px] mx-auto px-8 animate-fadeIn">
                <h1 className="text-center text-[#134252] text-4xl font-normal mb-3 mt-4">
                    🔬 Research Platform: Select Your Approach
                </h1>
                <p className="text-center text-[#5c6b72] text-base mb-8">
                    Choose between disease-first screening or plant-based discovery. Both paths integrate with BioDockify for advanced molecular analysis.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                    {/* PLAN A */}
                    <div
                        onClick={() => handlePlanSelect('A')}
                        className={`bg-white border-2 rounded-xl p-8 cursor-pointer transition-all duration-300 relative
                            ${selectedPlan === 'A' ? 'border-[#218081] bg-[rgba(33,128,141,0.02)]' : 'border-[#e0ddd7] hover:border-[#218081] hover:-translate-y-1 hover:shadow-xl'}
                        `}
                    >
                        <span className="inline-block bg-[#218081] text-white px-4 py-2 rounded-lg text-xs font-bold mb-4">
                            PLAN A
                        </span>
                        <div className="text-xl font-semibold mb-3 text-[#134252]">
                            🏥 Disease-First CNS Ensemble
                        </div>
                        <p className="text-[#5c6b72] text-sm mb-5 leading-relaxed">
                            Start with a known disease target and screen compounds against specific proteins. Ideal for rapid drug candidate ranking.
                        </p>
                        <ul className="list-none p-0 mb-5 space-y-2">
                            {[
                                "Alzheimer's-specific AI models (AChE, BACE1, GSK-3β)",
                                "SMILES input or library browsing",
                                "ChemBERTa + MolFormer ensemble scoring",
                                "Real-time binding predictions",
                                "ADMET & toxicity assessment",
                                "Publication-ready exports"
                            ].map((item, i) => (
                                <li key={i} className="text-[#5c6b72] text-sm py-2 border-b border-[#e0ddd7] last:border-0 flex items-center">
                                    <span className="text-[#209a66] font-bold mr-2">✓</span> {item}
                                </li>
                            ))}
                        </ul>

                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Select Disease:</label>
                            <select
                                className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white focus:outline-none focus:border-[#218081] focus:ring-2 focus:ring-[#218081]/10"
                                value={planADisease}
                                onChange={(e) => setPlanADisease(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <option value="">Choose a disease...</option>
                                <option value="alzheimers">Alzheimer's Disease (Primary)</option>
                                <option value="parkinsons">Parkinson's Disease</option>
                                <option value="als">Amyotrophic Lateral Sclerosis</option>
                                <option value="custom">Custom Targets</option>
                            </select>
                        </div>
                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Primary Target:</label>
                            <select
                                className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white focus:outline-none focus:border-[#218081] focus:ring-2 focus:ring-[#218081]/10"
                                value={planATarget}
                                onChange={(e) => setPlanATarget(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <option value="">Choose target...</option>
                                <option value="ache">AChE (Acetylcholinesterase)</option>
                                <option value="bace1">BACE1 (β-secretase 1)</option>
                                <option value="gsk3b">GSK-3β (Glycogen Synthase Kinase 3β)</option>
                                <option value="multi">Multi-target (all three)</option>
                            </select>
                        </div>
                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Compounds to Screen:</label>
                            <textarea
                                className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white min-h-[100px] resize-y focus:outline-none focus:border-[#218081] focus:ring-2 focus:ring-[#218081]/10"
                                placeholder="Paste SMILES strings (one per line)..."
                                value={planACompounds}
                                onChange={(e) => setPlanACompounds(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                            ></textarea>
                            <small className="text-[#5c6b72] block mt-1">Or: <a href="#" className="text-[#218081] hover:underline">Browse library (top 1000 AD leads)</a></small>
                        </div>

                        <button
                            className="w-full bg-[#218081] text-white py-3 px-6 rounded-lg font-bold text-sm hover:bg-[#1a6468] transition-colors flex items-center justify-center gap-2"
                            onClick={(e) => { e.stopPropagation(); startPlanA(); }}
                        >
                            → Start Disease Screening
                        </button>
                    </div>

                    {/* PLAN B */}
                    <div
                        onClick={() => handlePlanSelect('B')}
                        className={`bg-white border-2 rounded-xl p-8 cursor-pointer transition-all duration-300 relative
                            ${selectedPlan === 'B' ? 'border-[#218081] bg-[rgba(33,128,141,0.02)]' : 'border-[#e0ddd7] hover:border-[#218081] hover:-translate-y-1 hover:shadow-xl'}
                        `}
                    >
                        <span className="inline-block bg-[#218081] text-white px-4 py-2 rounded-lg text-xs font-bold mb-4">
                            PLAN B
                        </span>
                        <div className="text-xl font-semibold mb-3 text-[#134252]">
                            🌿 Plant-First Phytochemical Discovery
                        </div>
                        <p className="text-[#5c6b72] text-sm mb-5 leading-relaxed">
                            Upload plant extract data (GC-MS) and let AI predict phytochemicals, targets, and binding potential. Perfect for PhD research.
                        </p>
                        <ul className="list-none p-0 mb-5 space-y-2">
                            {[
                                "GC-MS data parsing & phytochemical identification",
                                "Multi-target prediction (AChE, BACE1, GSK-3β)",
                                "Automated docking screening workflow",
                                "PhD thesis chapter generation",
                                "Publication-ready figures & tables",
                                "Wet-lab validation integration"
                            ].map((item, i) => (
                                <li key={i} className="text-[#5c6b72] text-sm py-2 border-b border-[#e0ddd7] last:border-0 flex items-center">
                                    <span className="text-[#209a66] font-bold mr-2">✓</span> {item}
                                </li>
                            ))}
                        </ul>

                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Plant Species:</label>
                            <select
                                className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white focus:outline-none focus:border-[#218081] focus:ring-2 focus:ring-[#218081]/10"
                                value={planBPlant}
                                onChange={(e) => setPlanBPlant(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <option value="">Choose a plant...</option>
                                <option value="evolvulus">Evolvulus alsinoides (Dwarf Morning Glory)</option>
                                <option value="cordia">Cordia dichotoma (Lasura)</option>
                                <option value="withania">Withania somnifera (Ashwagandha)</option>
                                <option value="custom">Custom Plant</option>
                            </select>
                        </div>
                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Target Disease:</label>
                            <select
                                className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white focus:outline-none focus:border-[#218081] focus:ring-2 focus:ring-[#218081]/10"
                                value={planBDisease}
                                onChange={(e) => setPlanBDisease(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <option value="">Choose disease...</option>
                                <option value="alzheimers">Alzheimer's Disease</option>
                                <option value="parkinsons">Parkinson's Disease</option>
                                <option value="cns">General CNS Disorders</option>
                            </select>
                        </div>
                        <div className="mb-5">
                            <label className="block font-medium mb-2 text-sm text-[#134252]">Upload GC-MS Data:</label>
                            <div
                                className="border-2 border-dashed border-[#e0ddd7] rounded-lg p-5 text-center cursor-pointer hover:border-[#218081] hover:bg-[rgba(33,128,141,0.02)] transition-all"
                                onClick={(e) => { e.stopPropagation(); /* Trigger file input */ }}
                            >
                                <input type="file" className="hidden" accept=".cdf,.netcdf,.csv" />
                                <div className="text-[#5c6b72] text-sm">
                                    <strong>📁 Click to upload</strong> or drag and drop<br />
                                    <small>Supported: NetCDF, CDF, CSV (m/z, RT, intensity)</small>
                                </div>
                            </div>
                        </div>

                        <button
                            className="w-full bg-[#218081] text-white py-3 px-6 rounded-lg font-bold text-sm hover:bg-[#1a6468] transition-colors flex items-center justify-center gap-2"
                            onClick={(e) => { e.stopPropagation(); startPlanB(); }}
                        >
                            → Upload Plant Extract
                        </button>
                    </div>
                </div>

                <div className="flex justify-between items-center mt-8 pt-8 border-t border-[#e0ddd7] mb-12">
                    <div>
                        <p className="text-[#5c6b72] text-sm">After completing your screening, proceed to <strong>BioDockify.com</strong> for detailed molecular docking & MD simulations.</p>
                    </div>
                    <button
                        className="bg-transparent text-[#218081] border border-[#e0ddd7] py-3 px-6 rounded-lg font-bold text-sm hover:border-[#218081] hover:bg-[rgba(33,128,141,0.02)] transition-all"
                        onClick={goToFormulation}
                    >
                        Go to Formulation Module →
                    </button>
                </div>
            </div>
        </div>
    );
}
