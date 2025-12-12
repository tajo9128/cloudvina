import { useState } from 'react'

export default function InteractionTable({ interactions }) {
    const [activeTab, setActiveTab] = useState('hbond')

    if (!interactions) return null

    const hbonds = interactions.hydrogen_bonds || []
    const hydrophobic = interactions.hydrophobic_contacts || []

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
                <div>
                    <h3 className="font-bold text-slate-900">Interaction Analysis</h3>
                    <p className="text-sm text-slate-500 mt-1">
                        Protein-ligand contacts detected by geometric analysis
                    </p>
                </div>
                <div className="flex space-x-2">
                    <button
                        onClick={() => setActiveTab('hbond')}
                        className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors
                            ${activeTab === 'hbond'
                                ? 'bg-primary-100 text-primary-700'
                                : 'text-slate-600 hover:bg-slate-100'}`}
                    >
                        Hydrogen Bonds ({hbonds.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('hydrophobic')}
                        className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors
                            ${activeTab === 'hydrophobic'
                                ? 'bg-amber-100 text-amber-700'
                                : 'text-slate-600 hover:bg-slate-100'}`}
                    >
                        Hydrophobic ({hydrophobic.length})
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                <table className="min-w-full">
                    <thead className="bg-slate-50 border-b border-slate-200 sticky top-0">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Residue
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Distance (Å)
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Protein Atom
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Ligand Atom
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-slate-200">
                        {activeTab === 'hbond' ? (
                            hbonds.length > 0 ? (
                                hbonds.map((bond, idx) => (
                                    <tr key={idx} className="hover:bg-slate-50">
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                                                {bond.residue}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900 font-mono">
                                            {bond.distance}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                                            {bond.protein_atom}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                                            {bond.ligand_atom}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="4" className="px-6 py-8 text-center text-slate-500">
                                        No hydrogen bonds detected
                                    </td>
                                </tr>
                            )
                        ) : (
                            hydrophobic.length > 0 ? (
                                hydrophobic.map((contact, idx) => (
                                    <tr key={idx} className="hover:bg-slate-50">
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                                                {contact.residue}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900 font-mono">
                                            {contact.distance}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                                            {contact.protein_atom}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                                            {contact.ligand_atom}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="4" className="px-6 py-8 text-center text-slate-500">
                                        No hydrophobic contacts detected
                                    </td>
                                </tr>
                            )
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
