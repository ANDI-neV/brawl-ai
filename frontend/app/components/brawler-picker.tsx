import React, { useState, useMemo, useEffect } from 'react';
import BrawlerIcon from "./brawler-icon";
import { useBrawler } from './brawler-context';

interface BrawlerPickerProps {
  name: string;
}

interface TableHeaderProps {
  title: string;
  sortable?: boolean;
  sortKey?: string;
  currentSort: { key: string; direction: 'asc' | 'desc' };
  onSort: (key: string) => void;
}

const TableHeader: React.FC<TableHeaderProps> = ({ title, sortable = false, sortKey, currentSort, onSort }) => (
  <th scope="col" className="px-6 py-3">
    <div className="flex items-center">
      {title}
      {sortable && sortKey && (
        <button onClick={() => onSort(sortKey)} className="ml-1.5">
          {currentSort.key === sortKey && (
            <span className="ml-1">{currentSort.direction === 'asc' ? '↑' : '↓'}</span>
          )}
        </button>
      )}
    </div>
  </th>
);

interface TableRowProps {
  brawler: BrawlerPickerProps;
  score: number | null;
  pickrate: number | null;
  onClick: (brawler: BrawlerPickerProps) => void;
}

const TableRow: React.FC<TableRowProps> = ({ brawler, score, pickrate, onClick }) => (
  <tr className="border-b bg-gray-800 border-gray-700 cursor-pointer hover:bg-gray-600" onClick={() => onClick(brawler)}>
    <th scope="row" className="px-6 py-4 font-medium whitespace-nowrap dark:text-white">
      <BrawlerIcon brawler={brawler.name}/>
    </th>
    <td className="px-6 py-4 mx-auto items-center">{score !== null ? score.toFixed(3): 'N/A'}</td>
    <td className="px-6 py-4 mx-auto items-center">{pickrate !== null ? pickrate?.toFixed(3): 'N/A' }</td>
  </tr>
);

export default function BrawlerPicker() {
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: 'score', direction: 'desc' });
  const { selectBrawler, updatePredictions, availableBrawlers, selectedBrawlers, selectedMap, brawlerScores, brawlerPickrates } = useBrawler();
  const [localAvailableBrawlers, setLocalAvailableBrawlers] = useState<BrawlerPickerProps[]>(availableBrawlers);

  useEffect(() => {
    setLocalAvailableBrawlers(availableBrawlers);
  }, [availableBrawlers]);

  const isMapSelected = selectedMap !== "";

  const handleClick = (brawler: BrawlerPickerProps) => {
    const emptySlot = selectedBrawlers.findIndex(slot => slot === null);
    if (emptySlot !== -1) {
      selectBrawler(brawler, emptySlot);
      updatePredictions(selectedMap, selectedBrawlers.filter(Boolean).map(b => b!.name), true);
    } else {
      alert("All slots are filled. Clear a slot before selecting a new brawler.");
    }
  };

  const filterBrawlers = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilter(e.target.value);
  };

  const handleSort = (key: string) => {
    setSort(prevSort => ({
      key,
      direction: prevSort.key === key && prevSort.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const filteredAndSortedBrawlers = useMemo(() => {
    return localAvailableBrawlers
      .filter(brawler => brawler.name.toLowerCase().includes(filter.toLowerCase()))
      .sort((a, b) => {
        const scoreA = brawlerScores[a.name.toLowerCase()] ?? -Infinity;
        const scoreB = brawlerScores[b.name.toLowerCase()] ?? -Infinity;
        if (sort.key === 'score') {
          return sort.direction === 'desc' ? scoreB - scoreA : scoreA - scoreB;
        }
        return 0;
      });
  }, [localAvailableBrawlers, filter, sort, brawlerScores]);

  return (
    <div className="relative">
      <input
        type="text"
        placeholder="Filter brawlers..."
        onChange={filterBrawlers}
        className="w-full p-2 mb-4 border rounded-xl"
      />
      <div className="relative overflow-x-auto h-[500px] shadow-md rounded-xl bg-gray-800 custom-scrollbar">
        <div className="min-w-[250px]">
          <table className="w-full text-sm text-left rtl:text-right text-gray-400">
            <thead className="text-xs uppercase  bg-gray-700 text-gray-400">
              <tr>
                <TableHeader title="Brawler" sortable={false} currentSort={sort} onSort={handleSort} />
                <TableHeader title="Score" sortable sortKey="score" currentSort={sort} onSort={handleSort} />
                <TableHeader title="Pick Rate" sortable sortKey="pickRate" currentSort={sort} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {filteredAndSortedBrawlers.map((brawler) => (
                <TableRow 
                  key={brawler.name} 
                  brawler={brawler}
                  score={brawlerScores[brawler.name.toLowerCase()] ?? null}
                  pickrate={brawlerPickrates[brawler.name.toLowerCase()] ?? null}
                  onClick={handleClick}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {!isMapSelected && (
        <div className="absolute inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center rounded-xl">
          <span className="text-white font-bold text-2xl">Select Map</span>
        </div>
      )}
    </div>
  );
}