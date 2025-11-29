import { useMemo } from 'react'

export default function DockingResultsTable({ poses }) {
    if (!poses || poses.length === 0) {
        return (
            <div className="text-center py-8 text-slate-500">
                No docking poses available
            </div>
        )
    }

    // Helper to color-code affinity values
    const getAffinityColor = (affinity) => {
        if (affinity <= -9) return 'text-green-700 font-bold'
        if (affinity <= -7) return 'text-green-600'
        if (affinity <= -5) return 'text-amber-600'
        return 'text-slate-600'
    }

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 bg-slate-50 border-b border-slate-200">
                <h3 className="font-bold text-slate-900">Top Binding Poses</h3>
                <p className="text-sm text-slate-500 mt-1">
                    {poses.length} docking modes ranked by binding affinity
                </p>
            </div>

            <div className="overflow-x-auto">
                <table className="min-w-full">
                    <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Mode
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                Affinity (kcal/mol)
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                RMSD l.b.
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                RMSD u.b.
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-slate-200">
                        {poses.map((pose, index) => (
                            <tr key={pose.mode} className={index === 0 ? 'bg-primary-50/30' : 'hover:bg-slate-50'}>
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium text-slate-900">{pose.mode}</span>
                                        {index === 0 && (
                                            <span className="px-2 py-0.5 text-xs font-bold bg-primary-100 text-primary-700 rounded">
                                                BEST
                                            </span>
                                        )}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <span className={`text-sm font-semibold ${getAffinityColor(pose.affinity)}`}>
                                        {pose.affinity}
                                    </span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 font-mono">
                                    {pose.rmsd_lb.toFixed(3)}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 font-mono">
                                    {pose.rmsd_ub.toFixed(3)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="px-6 py-4 bg-slate-50 border-t border-slate-200">
                <div className="flex items-center justify-between text-xs text-slate-500">
                    <div>
                        <span className="font-semibold">RMSD:</span> Root Mean Square Deviation from best pose
                    </div>
                    <div>
                        <span className="font-semibold">Lower affinity</span> = stronger binding
                    </div>
                </div>
            </div>
        </div>
    )
}
