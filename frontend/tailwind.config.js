/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Playfair Display"', 'Georgia', 'serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"DM Mono"', 'monospace'],
      },
      colors: {
        noir: {
          950: '#080808',
          900: '#0f0f0f',
          800: '#1a1a1a',
          700: '#242424',
          600: '#2e2e2e',
        },
        amber: {
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
        },
        cream: '#f5f0e8',
      },
    },
  },
  plugins: [],
}
