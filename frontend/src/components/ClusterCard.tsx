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
      className={`group relative flex flex-col gap-3 p-4 rounded-sm border cursor-pointer transition-all duration-100 select-none ${
        selected
          ? 'border-accent shadow-sm'
          : 'border-border hover:border-borders hover:bg-surface2/50'
      }`}
      style={selected ? { background: `${color}0d`, borderColor: color } : { background: 'var(--color-surface)' }}>

      {/* Header row */}
      <div className="flex items-start gap-4">
        {/* Number badge */}
        <div className="shrink-0 w-9 h-9 rounded-sm flex items-center justify-center font-mono text-[15px] font-bold text-white mt-0.5"
          style={{ background: color }}>
          {index + 1}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-[16px] leading-tight text-ink break-words">{cluster.name}</h3>
          <div className="flex items-center gap-2.5 mt-1.5">
            <span className="font-mono text-[13px] text-faint">{cluster.size} pts</span>
            <span className="text-faint text-[13px]">·</span>
            <span className="font-mono text-[13px] text-faint">{pct}%</span>
          </div>
        </div>
        {/* Expand button */}
        <button
          onClick={e => { e.stopPropagation(); onExpand() }}
          className="shrink-0 font-mono text-[12px] font-bold tracking-wider uppercase px-2.5 py-1 rounded-sm border border-border text-faint opacity-0 group-hover:opacity-100 transition-all hover:border-borders hover:text-muted">
          view →
        </button>
      </div>

      {/* Description */}
      {cluster.description && (
        <p className="text-[14px] text-muted leading-relaxed line-clamp-2">{cluster.description}</p>
      )}

      {/* Size bar */}
      <div className="h-1.5 w-full rounded-full overflow-hidden" style={{ background: 'var(--color-surface3)' }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
      </div>

    </div>
  )
}
