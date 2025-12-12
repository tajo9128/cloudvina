import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Mock Progress Store using Zustand + Persist (LocalStorage)
export const useProgressStore = create(
    persist(
        (set, get) => ({
            // { courseId: { lessonId: true } }
            completedLessons: {},

            // Mark a lesson as complete
            markLessonComplete: (courseId, lessonId) => set((state) => ({
                completedLessons: {
                    ...state.completedLessons,
                    [courseId]: {
                        ...state.completedLessons[courseId],
                        [lessonId]: true
                    }
                }
            })),

            // Check if lesson is complete
            isLessonComplete: (courseId, lessonId) => {
                const courseProgress = get().completedLessons[courseId];
                return courseProgress ? !!courseProgress[lessonId] : false;
            },

            // Get progress percentage for a course
            getCourseProgress: (courseId, totalLessons) => {
                const courseProgress = get().completedLessons[courseId];
                if (!courseProgress) return 0;
                const completedCount = Object.keys(courseProgress).length;
                return Math.round((completedCount / totalLessons) * 100);
            }
        }),
        {
            name: 'biodockify-learning-progress', // unique name
        }
    )
);
