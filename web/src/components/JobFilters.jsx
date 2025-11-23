import { useState } from 'react'

export default function JobFilters({ onFilterChange }) {
    const [filters, setFilters] = useState({
        status: '',
        search: '',
        minAffinity: '',
        maxAffinity: ''
    })

    const handleChange = (key, value) => {
        const newFilters = { ...filters, [key]: value }
        setFilters(newFilters)
        onFilterChange(newFilters)
    }

    const handleReset = () => {
        const emptyFilters = {
            status: '',
            search: '',
            minAffinity: '',
            maxAffinity: ''
        }
        setFilters(emptyFilters)
        onFilterChange(emptyFilters)
    }

    return (
        <div className="bg-white p-4 rounded-lg shadow-sm mb-6 border border-gray-200">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-gray-900">Filter Jobs</h3>
                <button
                    onClick={handleReset}
                    className="text-sm text-purple-600 hover:text-purple-700 font-medium"
                >
                    Clear Filters
                </button>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Search Job ID
                    </label>
                    <input
                        type="text"
                        placeholder="Search..."
                        value={filters.search}
                        onChange={(e) => handleChange('search', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Status
                    </label>
                    <select
                        value={filters.status}
                        onChange={(e) => handleChange('status', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    >
                        <option value="">All Statuses</option>
                        <option value="SUCCEEDED">Succeeded</option>
                        <option value="FAILED">Failed</option>
                        <option value="RUNNING">Running</option>
                        <option value="SUBMITTED">Submitted</option>
                        <option value="PENDING">Pending</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Min Affinity (kcal/mol)
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        placeholder="e.g., -10"
                        value={filters.minAffinity}
                        onChange={(e) => handleChange('minAffinity', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Affinity (kcal/mol)
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        placeholder="e.g., 0"
                        value={filters.maxAffinity}
                        onChange={(e) => handleChange('maxAffinity', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                </div>
            </div>
        </div>
    )
}
