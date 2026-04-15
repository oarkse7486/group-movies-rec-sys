import React, { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || '/api'
const TMDB_IMAGE = 'https://image.tmdb.org/t/p/w300'
const STEP_LABELS = ['Group', 'Rate', 'Strategy', 'Results']
const MIN_RATINGS = 5

const AVATAR_COLORS = [
  '#f59e0b', '#10b981', '#3b82f6', '#ec4899',
  '#8b5cf6', '#f97316', '#06b6d4', '#84cc16',
  '#ef4444', '#a855f7',
]

const FALLBACK_MOVIES = [
  { movie_id: 1,    title: 'Toy Story (1995)',                   genres: ['Animation'],  tmdb_id: 862  },
  { movie_id: 2,    title: 'Jumanji (1995)',                     genres: ['Adventure'],  tmdb_id: 8844 },
  { movie_id: 32,   title: 'Twelve Monkeys (1995)',              genres: ['Sci-Fi'],     tmdb_id: 63   },
  { movie_id: 50,   title: 'The Usual Suspects (1995)',          genres: ['Crime'],      tmdb_id: 629  },
  { movie_id: 260,  title: 'Star Wars: A New Hope (1977)',       genres: ['Sci-Fi'],     tmdb_id: 11   },
  { movie_id: 296,  title: 'Pulp Fiction (1994)',                genres: ['Crime'],      tmdb_id: 680  },
  { movie_id: 318,  title: 'The Shawshank Redemption (1994)',    genres: ['Drama'],      tmdb_id: 278  },
  { movie_id: 527,  title: "Schindler's List (1993)",            genres: ['Drama'],      tmdb_id: 424  },
  { movie_id: 589,  title: 'Terminator 2 (1991)',                genres: ['Action'],     tmdb_id: 280  },
  { movie_id: 858,  title: 'The Godfather (1972)',               genres: ['Crime'],      tmdb_id: 238  },
  { movie_id: 1196, title: 'The Empire Strikes Back (1980)',     genres: ['Sci-Fi'],     tmdb_id: 1891 },
  { movie_id: 1265, title: 'Groundhog Day (1993)',               genres: ['Comedy'],     tmdb_id: 137  },
  { movie_id: 2571, title: 'The Matrix (1999)',                  genres: ['Sci-Fi'],     tmdb_id: 603  },
  { movie_id: 2959, title: 'Fight Club (1999)',                  genres: ['Drama'],      tmdb_id: 550  },
  { movie_id: 4993, title: 'Lord of the Rings: Fellowship (2001)', genres: ['Fantasy'], tmdb_id: 120  },
  { movie_id: 5952, title: 'Catch Me If You Can (2002)',         genres: ['Drama'],      tmdb_id: 637  },
  { movie_id: 6539, title: 'Pirates of the Caribbean (2003)',    genres: ['Action'],     tmdb_id: 22   },
  { movie_id: 7153, title: 'The Lord of the Rings: Return (2003)', genres: ['Fantasy'], tmdb_id: 122  },
  { movie_id: 8961, title: 'The Incredibles (2004)',             genres: ['Animation'],  tmdb_id: 9806 },
  { movie_id: 58559, title: 'The Dark Knight (2008)',            genres: ['Action'],     tmdb_id: 155  },
]

/**
 * StarRating component - 1-5 star interactive rating control.
 */
function StarRating({ value, onChange }) {
  const [hover, setHover] = useState(0)
  const display = hover || value

  return (
    <div className="flex gap-1 justify-center mt-2">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
          onClick={() => onChange(star)}
          className="star-btn text-xl leading-none"
          style={{ color: star <= display ? '#f59e0b' : '#2e2e2e' }}
        >
          &#9733;
        </button>
      ))}
    </div>
  )
}

/**
 * RateMovies page.
 * Each group member rates at least MIN_RATINGS movies from a curated list.
 * Member index is taken from the URL param.
 */
export default function RateMovies({ members, setMembers, groupSize }) {
  const navigate = useNavigate()
  const { memberIndex } = useParams()
  const idx = parseInt(memberIndex, 10)
  const member = members[idx]

  const [movies, setMovies] = useState(FALLBACK_MOVIES)
  const [ratings, setRatings] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios
      .get(`${API}/movies/popular?limit=20`)
      .then((res) => { if (res.data?.length) setMovies(res.data) })
      .catch(() => { /* use fallback silently */ })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (member?.ratings) setRatings(member.ratings)
  }, [member])

  const handleRate = (movieId, score) => {
    setRatings((prev) => ({ ...prev, [movieId]: score }))
  }

  const saveAndNavigate = (destination) => {
    setMembers((prev) =>
      prev.map((m, i) => (i === idx ? { ...m, ratings } : m))
    )
    navigate(destination)
  }

  const ratedCount = Object.keys(ratings).length
  const canContinue = ratedCount >= MIN_RATINGS
  const nextPath = idx + 1 < groupSize ? `/rate/${idx + 1}` : '/strategy'

  if (!member) {
    navigate('/')
    return null
  }

  return (
    <div className="min-h-screen bg-noir-950 flex flex-col">

      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-6 border-b border-white/5">
        <button
          onClick={() => navigate('/')}
          className="font-display text-xl text-amber-400 tracking-wider"
          style={{ letterSpacing: '0.15em' }}
        >
          GROUP REC
        </button>
        <div className="flex items-center gap-2">
          {STEP_LABELS.map((step, i) => (
            <React.Fragment key={step}>
              <span className={`font-mono text-xs uppercase tracking-widest ${i === 1 ? 'text-amber-400' : 'text-white/20'}`}>
                {step}
              </span>
              {i < STEP_LABELS.length - 1 && (
                <span className="text-white/10 text-xs">&rsaquo;</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </nav>

      <main className="flex-1 px-6 py-12 max-w-5xl mx-auto w-full">

        {/* Member header */}
        <div className="flex items-center gap-4 mb-8 fade-up">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center font-display text-noir-950 font-bold text-xl flex-shrink-0"
            style={{ backgroundColor: AVATAR_COLORS[idx] }}
          >
            {member.name[0]?.toUpperCase()}
          </div>
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-white/40">
              Step 2 of 4 &middot; Member {idx + 1} of {groupSize}
            </p>
            <h2 className="font-display text-3xl text-cream">
              {member.name}'s turn
            </h2>
          </div>
        </div>

        <p className="font-body text-white/40 mb-2 fade-up fade-up-delay-1">
          Rate at least {MIN_RATINGS} movies. The more you rate, the better the recommendations.
        </p>

        {/* Progress bar */}
        <div className="w-full h-1 bg-noir-700 rounded-full mb-10 fade-up fade-up-delay-1">
          <div
            className="h-1 bg-amber-400 rounded-full transition-all duration-500"
            style={{ width: `${Math.min((ratedCount / MIN_RATINGS) * 100, 100)}%` }}
          />
        </div>

        {/* Movie grid */}
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <p className="font-mono text-white/30 text-sm animate-pulse">
              Loading films...
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-12 fade-up fade-up-delay-2">
            {movies.map((movie) => {
              const rated = ratings[movie.movie_id]
              return (
                <div
                  key={movie.movie_id}
                  className="relative rounded-xl overflow-hidden border transition-all duration-200"
                  style={{
                    background: '#1a1a1a',
                    borderColor: rated ? 'rgba(245,158,11,0.6)' : 'rgba(255,255,255,0.05)',
                    boxShadow: rated ? '0 0 16px rgba(245,158,11,0.15)' : 'none',
                  }}
                >
                  {/* Poster */}
                  {movie.tmdb_id ? (
                    <img
                      src={`${TMDB_IMAGE}${movie.tmdb_id}`}
                      alt={movie.title}
                      className="w-full aspect-[2/3] object-cover"
                      onError={(e) => { e.target.style.display = 'none' }}
                    />
                  ) : (
                    <div className="w-full aspect-[2/3] bg-noir-700 flex items-center justify-center">
                      <span className="font-display text-white/10 text-4xl">[ ]</span>
                    </div>
                  )}

                  {/* Rated badge */}
                  {rated && (
                    <div
                      className="absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center font-mono text-xs font-bold text-noir-950"
                      style={{ backgroundColor: '#f59e0b' }}
                    >
                      {rated}
                    </div>
                  )}

                  <div className="p-3">
                    <p className="font-body text-white/80 text-xs leading-tight mb-1 line-clamp-2">
                      {movie.title}
                    </p>
                    <p className="font-mono text-white/30 text-xs mb-1">
                      {movie.genres?.[0]}
                    </p>
                    <StarRating
                      value={rated || 0}
                      onChange={(score) => handleRate(movie.movie_id, score)}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Bottom bar */}
        <div className="flex items-center justify-between">
          <span className="font-mono text-sm text-white/30">
            {ratedCount} rated
            {!canContinue && (
              <span style={{ color: 'rgba(245,158,11,0.6)' }}>
                {' '}({MIN_RATINGS - ratedCount} more needed)
              </span>
            )}
          </span>

          <div className="flex gap-3 items-center">
            {ratedCount > 0 && !canContinue && (
              <button
                onClick={() => saveAndNavigate(nextPath)}
                className="font-body text-white/30 hover:text-white/60 text-sm px-6 py-3 transition-colors"
              >
                Skip for now
              </button>
            )}
            <button
              onClick={() => saveAndNavigate(nextPath)}
              disabled={!canContinue}
              className="font-body font-semibold text-lg px-10 py-3 rounded-full transition-all duration-300"
              style={{
                backgroundColor: canContinue ? '#f59e0b' : '#242424',
                color: canContinue ? '#080808' : 'rgba(255,255,255,0.2)',
                cursor: canContinue ? 'pointer' : 'not-allowed',
              }}
              onMouseEnter={(e) => {
                if (canContinue) e.currentTarget.style.boxShadow = '0 0 30px rgba(245,158,11,0.4)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              {idx + 1 < groupSize
                ? `Next: ${members[idx + 1]?.name}`
                : 'Choose Strategy'} &rarr;
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
