import React from 'react'
import type { Cluster } from '../types'

const CLUSTER_COLORS = [
  '#4a7c59', '#6a7a3a', '#3a6a7a', '#7a5a3a', '#5a4a7a',
  '#3a7a6a', '#7a3a5a', '#4a5a7a', '#7a6a3a', '#3a5a4a',
]

interface Props {
  cluster: Cluster
  index: number
  selected: boolean
  totalSize: number
  onSelect: () => void
  onExpand: () => void
}

export default function ClusterCard({ cluster, index, selected, totalSize, onSelect, onExpand }: Props) {
  const color = CLUSTER_COLORS[index % CLUSTER_COLORS.length]
  const pct = totalSize > 0 ? Math.round((cluster.size / totalSize) * 100) : 0

  return (
    <div
      onClick={onSelect}
      className={`group relative flex flex-col gap-4 p-6 rounded-sm border cursor-pointer transition-all duration-100 select-none ${
        selected
          ? 'border-accent shadow-sm'
          : 'border-border hover:border-borders hover:bg-surface2/50'
      }`}
      style={selected ? { background: `${color}0d`, borderColor: color } : { background: 'var(--color-surface)' }}>

      {/* Header row */}
      <div className="flex items-start gap-4">
        {/* Number badge */}
        <div className="shrink-0 w-11 h-11 rounded-sm flex items-center justify-center font-mono text-[18px] font-bold text-white mt-0.5"
          style={{ background: color }}>
          {index + 1}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-[20px] leading-tight text-ink truncate">{cluster.name}</h3>
          <div className="flex items-center gap-2.5 mt-2">
            <span className="font-mono text-[15px] text-faint">{cluster.size} pts</span>
            <span className="text-faint text-[15px]">·</span>
            <span className="font-mono text-[15px] text-faint">{pct}%</span>
          </div>
        </div>
        {/* Expand button */}
        <button
          onClick={e => { e.stopPropagation(); onExpand() }}
          className="shrink-0 font-mono text-[14px] font-bold tracking-wider uppercase px-3 py-1.5 rounded-sm border border-border text-faint opacity-0 group-hover:opacity-100 transition-all hover:border-borders hover:text-muted">
          view →
        </button>
      </div>

      {/* Description */}
      {cluster.description && (
        <p className="text-[17px] text-muted leading-relaxed line-clamp-2">{cluster.description}</p>
      )}

      {/* Size bar */}
      <div className="h-1.5 w-full rounded-full overflow-hidden" style={{ background: 'var(--color-surface3)' }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
      </div>

      {/* Selected indicator */}
      {selected && (
        <div className="absolute top-3.5 right-3.5 w-3 h-3 rounded-full" style={{ background: color }} />
      )}
    </div>
  )
}
