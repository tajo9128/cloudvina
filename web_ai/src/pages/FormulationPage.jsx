import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function FormulationPage() {
    const navigate = useNavigate();
    const [activeStep, setActiveStep] = useState(1);

    // Simple state containers for demo
    const [dosageForm, setDosageForm] = useState('tablet-ir');

    const steps = [
        { id: 1, label: "Lead Compound Input" },
        { id: 2, label: "Dosage Form" },
        { id: 3, label: "Target Profile (TPP)" },
        { id: 4, label: "Excipient AI" },
        { id: 5, label: "Manufacturing" },
        { id: 6, label: "Stability" },
        { id: 7, label: "Compatibility" },
        { id: 8, label: "Cost & Timeline" },
        { id: 9, label: "Report Export" },
        { id: 10, label: "BioDockify Link" }
    ];

    const switchComponent = (stepId) => {
        setActiveStep(stepId);
    };

    return (
        <div className="bg-[#f9faf8] min-h-screen text-[#134252] font-sans p-8">
            <div className="max-w-[1400px] mx-auto">
                <h1 className="text-3xl font-normal mb-3">⚗️ AI Formulation Optimization</h1>
                <p className="text-[#5c6b72] mb-8">Transform your lead compounds (from BioDockify docking) into manufacturing-ready pharmaceutical formulations</p>

                <div className="grid grid-cols-1 lg:grid-cols-[250px_1fr_300px] gap-6">
                    {/* LEFT SIDEBAR: COMPONENTS */}
                    <div className="rounded-xl border border-[#e0ddd7] bg-white p-4 h-fit sticky top-5">
                        <div className="mb-4">
                            <div className="text-xs font-semibold uppercase text-[#5c6b72] mb-3 tracking-wider">Components</div>
                            {steps.map(step => (
                                <div
                                    key={step.id}
                                    onClick={() => switchComponent(step.id)}
                                    className={`px-3 py-2 rounded-lg cursor-pointer text-sm transition-all border-l-[3px] mb-1
                                        ${activeStep === step.id
                                            ? 'bg-[rgba(33,128,141,0.1)] border-[#218081] font-semibold'
                                            : 'border-transparent hover:bg-[rgba(33,128,141,0.05)] hover:border-[#218081]'}
                                    `}
                                >
                                    {step.id === 10 ? '⑩' : `①②③④⑤⑥⑦⑧⑨`.charAt(step.id - 1)} {step.label}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* MAIN CONTENT */}
                    <div className="rounded-xl border border-[#e0ddd7] bg-white p-8 min-h-[500px]">

                        {/* Component 1: Lead Input */}
                        {activeStep === 1 && (
                            <div className="animate-fadeIn">
                                <div className="bg-[rgba(33,128,141,0.1)] px-3 py-2 rounded text-xs text-[#218081] font-bold mb-4 w-fit">Step 1 of 10</div>
                                <h2 className="text-lg font-semibold mb-5 text-[#134252]">① Lead Compound Input</h2>
                                <p className="mb-5 text-[#5c6b72]">Select or upload your lead compounds from the BioDockify screening results.</p>

                                <div className="mb-6">
                                    <label className="block font-medium mb-2 text-sm">Method 1: Select from Previous Results</label>
                                    <table className="w-full text-sm border-collapse mb-5">
                                        <thead>
                                            <tr className="bg-[rgba(33,128,141,0.05)] border-b border-[#e0ddd7]">
                                                <th className="p-3 text-left font-semibold text-[#134252]">Compound</th>
                                                <th className="p-3 text-left font-semibold text-[#134252]">Binding Energy</th>
                                                <th className="p-3 text-left font-semibold text-[#134252]">Target</th>
                                                <th className="p-3 text-left font-semibold text-[#134252]">Status</th>
                                                <th className="p-3 text-left font-semibold text-[#134252]">Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr className="border-b border-[#e0ddd7] hover:bg-[rgba(33,128,141,0.02)]">
                                                <td className="p-3">Scopoletin</td>
                                                <td className="p-3">-8.9 kcal/mol</td>
                                                <td className="p-3">AChE</td>
                                                <td className="p-3"><span className="bg-[rgba(32,154,102,0.1)] text-[#209a66] px-2 py-1 rounded text-xs font-bold">✓ PASS</span></td>
                                                <td className="p-3"><input type="checkbox" /></td>
                                            </tr>
                                            <tr className="border-b border-[#e0ddd7] hover:bg-[rgba(33,128,141,0.02)]">
                                                <td className="p-3">Quercetin</td>
                                                <td className="p-3">-8.5 kcal/mol</td>
                                                <td className="p-3">GSK-3β</td>
                                                <td className="p-3"><span className="bg-[rgba(32,154,102,0.1)] text-[#209a66] px-2 py-1 rounded text-xs font-bold">✓ PASS</span></td>
                                                <td className="p-3"><input type="checkbox" /></td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                <button
                                    className="w-full bg-[#218081] text-white py-3 px-6 rounded-lg font-bold text-sm hover:bg-[#1a6468] transition-colors"
                                    onClick={() => switchComponent(2)}
                                >
                                    Next: Dosage Form →
                                </button>
                            </div>
                        )}

                        {/* Component 2: Dosage Form */}
                        {activeStep === 2 && (
                            <div className="animate-fadeIn">
                                <div className="bg-[rgba(33,128,141,0.1)] px-3 py-2 rounded text-xs text-[#218081] font-bold mb-4 w-fit">Step 2 of 10</div>
                                <h2 className="text-lg font-semibold mb-5 text-[#134252]">② Dosage Form Selection</h2>
                                <p className="mb-5 text-[#5c6b72]">Choose your preferred pharmaceutical formulation type.</p>

                                <div className="mb-6">
                                    <label className="block font-medium mb-2 text-sm">Primary Dosage Form:</label>
                                    <select
                                        className="w-full p-3 border border-[#e0ddd7] rounded-lg text-sm bg-white focus:outline-none focus:border-[#218081]"
                                        value={dosageForm}
                                        onChange={(e) => setDosageForm(e.target.value)}
                                    >
                                        <option value="tablet-ir">Tablet - Immediate Release (IR)</option>
                                        <option value="tablet-mr">Tablet - Modified Release (MR)</option>
                                        <option value="capsule">Capsule (Hard/Soft Gelatin)</option>
                                        <option value="solution">Oral Solution/Suspension</option>
                                    </select>
                                </div>

                                <div className="bg-[rgba(33,128,141,0.05)] border-l-4 border-[#218081] p-4 rounded mb-6 text-sm">
                                    <strong className="text-[#218081]">💡 Recommendation:</strong> Oral tablet (immediate release) is recommended for Alzheimer's disease drugs. Suitable for multi-target phytochemicals.
                                </div>

                                <div className="flex justify-between">
                                    <button className="text-[#218081] border border-[#e0ddd7] py-3 px-6 rounded-lg font-bold text-sm hover:bg-gray-50" onClick={() => switchComponent(1)}>← Back</button>
                                    <button className="bg-[#218081] text-white py-3 px-6 rounded-lg font-bold text-sm hover:bg-[#1a6468]" onClick={() => switchComponent(3)}>Next: Target Profile →</button>
                                </div>
                            </div>
                        )}

                        {/* Simple Skeleton for other steps */}
                        {activeStep > 2 && activeStep < 10 && (
                            <div className="animate-fadeIn">
                                <div className="bg-[rgba(33,128,141,0.1)] px-3 py-2 rounded text-xs text-[#218081] font-bold mb-4 w-fit">Step {activeStep} of 10</div>
                                <h2 className="text-lg font-semibold mb-5 text-[#134252]">Component {activeStep} Content</h2>
                                <p className="text-[#5c6b72]">Placeholder functionality for step {activeStep}.</p>
                                <div className="flex justify-between mt-8">
                                    <button className="text-[#218081] border border-[#e0ddd7] py-3 px-6 rounded-lg font-bold text-sm hover:bg-gray-50" onClick={() => switchComponent(activeStep - 1)}>← Back</button>
                                    <button className="bg-[#218081] text-white py-3 px-6 rounded-lg font-bold text-sm hover:bg-[#1a6468]" onClick={() => switchComponent(activeStep + 1)}>Next →</button>
                                </div>
                            </div>
                        )}

                        {/* Step 10: Exit */}
                        {activeStep === 10 && (
                            <div className="animate-fadeIn text-center py-12">
                                <h2 className="text-2xl font-semibold mb-4 text-[#134252]">Ready for Production?</h2>
                                <p className="text-[#5c6b72] mb-8">Return to main dashboard to finalize.</p>
                                <button className="bg-[#218081] text-white py-3 px-8 rounded-lg font-bold hover:bg-[#1a6468]" onClick={() => window.location.href = 'https://biodockify.com'}>Go to BioDockify Main</button>
                            </div>
                        )}
                    </div>

                    {/* RIGHT PANEL */}
                    <div className="rounded-xl border border-[#e0ddd7] bg-white p-5 h-fit sticky top-5">
                        <div className="font-semibold mb-4 text-sm text-[#134252]">Formulation Preview</div>
                        <div className="border-b border-[#e0ddd7] py-2 flex justify-between text-xs">
                            <span className="text-[#5c6b72]">Selected Compound</span>
                            <span className="font-semibold text-[#134252]">Scopoletin</span>
                        </div>
                        <div className="border-b border-[#e0ddd7] py-2 flex justify-between text-xs">
                            <span className="text-[#5c6b72]">Dosage Form</span>
                            <span className="font-semibold text-[#134252]">{dosageForm === 'tablet-ir' ? 'Tablet (IR)' : dosageForm}</span>
                        </div>
                        <div className="border-b border-[#e0ddd7] py-2 flex justify-between text-xs">
                            <span className="text-[#5c6b72]">Total Excipients</span>
                            <span className="font-semibold text-[#134252]">4 (Predicted)</span>
                        </div>
                        <div className="border-b border-0 py-2 flex justify-between text-xs">
                            <span className="text-[#5c6b72]">Est. Cost/Unit</span>
                            <span className="font-semibold text-[#134252]">$0.12</span>
                        </div>
                        <button className="w-full mt-4 bg-white border border-[#e0ddd7] text-[#218081] py-2 rounded text-xs font-bold hover:bg-gray-50">📥 Export PDF Summary</button>
                    </div>

                </div>
            </div>
        </div>
    );
}
