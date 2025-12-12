import React from 'react';

export default function CompoundPreview({ compounds }) {
    return (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50">
                    <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">SMILES</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Props</th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-200">
                    {compounds.map((c, i) => (
                        <tr key={i}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-600 max-w-xs truncate">{c.smiles}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900">{c.chem_name}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-xs text-slate-500">
                                {Object.keys(c.properties).filter(k => k !== 'smiles').slice(0, 3).join(', ')}
                            </td>
                        </tr>
                    ))}
                    {compounds.length === 0 && (
                        <tr><td colSpan="3" className="px-6 py-8 text-center text-slate-500">No data uploaded yet.</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}
