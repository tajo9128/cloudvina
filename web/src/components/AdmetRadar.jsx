import React from 'react';

export default function AdmetRadar({ data, width = 300, height = 300 }) {
    if (!data) return null;

    // Normalize Data for Radar Chart (Goal: 1.0 is "Limit", Inner is Safe)
    // Lipinski Rules: MW < 500, LogP < 5, HBD < 5, HBA < 10, TPSA < 140 (Veber)
    // We normalize so that "1.0" represents the Threshold.

    const axes = [
        { name: 'MW', value: data.molecular_properties.molecular_weight, max: 500, label: 'Size' },
        { name: 'LogP', value: data.molecular_properties.logp, max: 5, label: 'Lipophilicity' },
        { name: 'HBD', value: data.molecular_properties.hbd, max: 5, label: 'H-Donors' },
        { name: 'HBA', value: data.molecular_properties.hba, max: 10, label: 'H-Acceptors' },
        { name: 'TPSA', value: data.molecular_properties.tpsa, max: 140, label: 'Polarity' },
    ];

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 40; // Padding

    // Helper to get coordinates
    const getCoordinates = (value, max, index, total) => {
        const angle = (Math.PI * 2 * index) / total - Math.PI / 2; // Start at top
        // Cap value at 1.5x limit for visual containment
        const normalized = Math.min(Math.max(value, 0), max * 1.5) / (max * 1.5);
        // Scale: Threshold (max) should be at 66% (2/3) of radius
        const r = (value / max) * (radius * 0.66);

        return {
            x: centerX + Math.cos(angle) * r,
            y: centerY + Math.sin(angle) * r,
            labelX: centerX + Math.cos(angle) * (radius + 20),
            labelY: centerY + Math.sin(angle) * (radius + 20)
        };
    };

    // Generate Polygon Points
    const points = axes.map((axis, i) => {
        const { x, y } = getCoordinates(axis.value, axis.max, i, axes.length);
        return `${x},${y}`;
    }).join(' ');

    // Generate Threshold Polygon (The "Safe Zone")
    const thresholdPoints = axes.map((axis, i) => {
        const angle = (Math.PI * 2 * i) / axes.length - Math.PI / 2;
        const r = radius * 0.66; // The limit line
        return `${centerX + Math.cos(angle) * r},${centerY + Math.sin(angle) * r}`;
    }).join(' ');

    return (
        <div className="relative flex flex-col items-center">
            <svg
                viewBox={`0 0 ${width} ${height}`}
                className="overflow-visible w-full h-auto max-w-[350px]"
            >
                {/* Background Grid Circles */}
                {[0.33, 0.66, 1].map((scale, i) => (
                    <circle
                        key={i}
                        cx={centerX}
                        cy={centerY}
                        r={radius * scale}
                        fill="none"
                        stroke={i === 1 ? "#ef4444" : "#e2e8f0"} // Red line for Limit
                        strokeWidth={i === 1 ? 2 : 1}
                        strokeDasharray={i === 1 ? "4 4" : "0"}
                        className="opacity-50"
                    />
                ))}

                {/* Axes Lines */}
                {axes.map((axis, i) => {
                    const angle = (Math.PI * 2 * i) / axes.length - Math.PI / 2;
                    const x2 = centerX + Math.cos(angle) * radius;
                    const y2 = centerY + Math.sin(angle) * radius;
                    return (
                        <line key={`line-${i}`} x1={centerX} y1={centerY} x2={x2} y2={y2} stroke="#cbd5e1" strokeWidth="1" />
                    );
                })}

                {/* The Data Polygon */}
                <polygon points={points} fill="rgba(99, 102, 241, 0.2)" stroke="#6366f1" strokeWidth="2" />

                {/* Data Points & Labels */}
                {axes.map((axis, i) => {
                    const { x, y, labelX, labelY } = getCoordinates(axis.value, axis.max, i, axes.length);
                    const isViolation = axis.value > axis.max;
                    return (
                        <g key={`point-${i}`}>
                            <circle cx={x} cy={y} r="4" fill={isViolation ? "#ef4444" : "#6366f1"} />
                            <text
                                x={labelX}
                                y={labelY}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                className={`text-[10px] font-medium ${isViolation ? 'fill-red-500 font-bold' : 'fill-slate-500'}`}
                            >
                                {axis.label}
                                <tspan x={labelX} dy="1.2em" className="fill-slate-900 font-bold">{axis.value}</tspan>
                            </text>
                        </g>
                    );
                })}
            </svg>
            <div className="mt-2 text-xs text-slate-400 flex items-center gap-2">
                <span className="w-3 h-0.5 bg-red-400 border-dashed border-red-400"></span> Limit (Rule of 5)
            </div>
        </div>
    );
}
