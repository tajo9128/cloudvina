/** @type {import('tailwindcss').Config} */
export default {
    content: [
        700: '#1e3a8a',
        800: '#1e3a8a',
        900: '#1e40af',
                },
'tech-cyan': {
    50: '#ecfeff',
        100: '#cffafe',
            200: '#a5f3fc',
                300: '#67e8f9',
                    400: '#22d3ee',
                        500: '#06b6d4',
                            600: '#0891b2',
                                700: '#0e7490',
                                    800: '#155e75',
                                        900: '#164e63',
                },
            },
backgroundImage: {
    'blue-gradient': 'linear-gradient(to bottom right, #1e3a8a, #0a1e3e)',
        'blue-gradient-radial': 'radial-gradient(circle, #3b82f6, #1e3a8a)',
            'tech-mesh': 'linear-gradient(135deg, #0a1e3e 0%, #1e3a8a 25%, #3b82f6 50%, #1e3a8a 75%, #0a1e3e 100%)',
            },
boxShadow: {
    'blue-glow': '0 0 20px rgba(59, 130, 246, 0.5)',
        'blue-glow-lg': '0 0 40px rgba(59, 130, 246, 0.6)',
            'deep-blue': '0 10px 40px rgba(10, 30, 62, 0.3)',
            },
animation: {
    'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
            },
keyframes: {
    float: {
        '0%, 100%': { transform: 'translateY(0px)' },
        '50%': { transform: 'translateY(-20px)' },
    },
},
        },
    },
plugins: [],
}
