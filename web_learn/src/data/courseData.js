// Comprehensive mock data for the Course System (LearnDash style)

export const courses = [
    {
        id: 'c1',
        title: 'Molecular Docking Fundamentals',
        slug: 'molecular-docking-fundamentals',
        description: 'Master the basics of molecular docking with AutoDock Vina. Learn protein preparation, ligand preparation, and grid box generation.',
        thumbnail: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000',
        instructor: {
            name: 'Dr. Sarah Smith',
            avatar: 'https://randomuser.me/api/portraits/women/44.jpg',
            role: 'Lead Scientist'
        },
        level: 'Beginner',
        duration: '4 hours',
        lessonsCount: 12,
        students: 1540,
        rating: 4.8,
        price: 'Free',
        tags: ['AutoDock Vina', 'Basics', 'Protein Prep'],
        curriculum: [
            {
                id: 'm1',
                title: 'Introduction to Docking',
                lessons: [
                    { id: 'l1', title: 'What is Molecular Docking?', duration: '10:00', type: 'video', isFree: true },
                    { id: 'l2', title: 'The Lock and Key Principle', duration: '15:00', type: 'video', isFree: true },
                    { id: 'q1', title: 'Intro Quiz', duration: '5:00', type: 'quiz', isFree: true }
                ]
            },
            {
                id: 'm2',
                title: 'Preparing Your Files',
                lessons: [
                    { id: 'l3', title: 'Protein Preparation Checklist', duration: '20:00', type: 'video', isFree: false },
                    { id: 'l4', title: 'Ligand Optimization', duration: '25:00', type: 'video', isFree: false },
                    { id: 'l5', title: 'Defining the Grid Box', duration: '18:00', type: 'video', isFree: false }
                ]
            }
        ]
    },
    {
        id: 'c2',
        title: 'Advanced Virtual Screening Strategies',
        slug: 'advanced-virtual-screening',
        description: 'Learn how to screen millions of compounds efficiently. Covers HTVS, consensus scoring, and post-processing with Python.',
        thumbnail: 'https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?auto=format&fit=crop&q=80&w=1000',
        instructor: {
            name: 'Dr. Alex Chen',
            avatar: 'https://randomuser.me/api/portraits/men/32.jpg',
            role: 'Computational Chemist'
        },
        level: 'Advanced',
        duration: '8 hours',
        lessonsCount: 20,
        students: 850,
        rating: 4.9,
        price: '$49.99',
        tags: ['Virtual Screening', 'Python', 'Big Data'],
        curriculum: [
            {
                id: 'm1',
                title: 'Virtual Screening Workflow',
                lessons: [
                    { id: 'l1', title: 'Designing a Screening Funnel', duration: '30:00', type: 'video', isFree: true },
                    { id: 'l2', title: 'Library Preparation', duration: '45:00', type: 'video', isFree: false }
                ]
            }
        ]
    },
    {
        id: 'c3',
        title: 'Python for Drug Discovery',
        slug: 'python-for-drug-discovery',
        description: 'Automate your docking workflows with Python. Learn RDKit, Pandas, and PyMOL scripting.',
        thumbnail: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&q=80&w=1000',
        instructor: {
            name: 'Mike Johnson',
            avatar: 'https://randomuser.me/api/portraits/men/86.jpg',
            role: 'Bioinformatics Engineer'
        },
        level: 'Intermediate',
        duration: '6 hours',
        lessonsCount: 15,
        students: 2100,
        rating: 4.7,
        price: '$29.99',
        tags: ['Python', 'RDKit', 'Automation'],
        curriculum: []
    }
];

export const categories = [
    { id: 'all', name: 'All Courses' },
    { id: 'docking', name: 'Molecular Docking' },
    { id: 'screening', name: 'Virtual Screening' },
    { id: 'md', name: 'Molecular Dynamics' },
    { id: 'informatics', name: 'Cheminformatics' }
];
