import React, { useState, useMemo, useEffect } from 'react';
import BrawlerIcon from "./brawler-icon";
import { useBrawler } from './brawler-context';
import { motion } from 'framer-motion';

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
  <th scope="col" className="px-6 py-3 text-center">
    <motion.button  onClick={() => sortable && onSort(sortKey)} className="flex items-center justify-center w-full"
      whileHover={{ scale: 1.1, zIndex: 10 }}
      whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}>
      {title}
      {sortable && sortKey && (
        <div className="ml-1.5">
          {currentSort.key != sortKey && (
            <span className="ml-1">{currentSort.direction === 'asc' ? '↑' : '↓'}</span>
          )}
          {currentSort.key === sortKey && (
            <span className="ml-1 text-white">{currentSort.direction === 'asc' ? '↑' : '↓'}</span>
          )}
          
        </div>
      )}
    </motion.button>
  </th>
);

interface TableRowProps {
  brawler: BrawlerPickerProps;
  score: number | null;
  pickrate: number | null;
  onClick: (brawler: BrawlerPickerProps) => void;
  onRightClick: (brawler: BrawlerPickerProps) => void;
  disabled: boolean;
  isAboveFold: boolean;
}

const TableRow: React.FC<TableRowProps> = ({ brawler, score, pickrate, onClick, onRightClick, disabled, isAboveFold}) => (
  <tr 
    className={`border-b bg-gray-800 border-gray-700 ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-gray-600'}`} 
    onClick={() => !disabled && onClick(brawler)}
    onContextMenu={(e) => {!disabled && onRightClick(brawler); e.preventDefault()}}
  >
    <th scope="row" className="px-2 md:px-4 py-1 font-medium text-white text-center flex items-center justify-center">
      <BrawlerIcon 
        brawler={brawler.name} 
        isAboveFold={isAboveFold}/>
    </th>
    <td className="px-2 py-4 mx-auto items-center text-md text-center">{score !== null ? score.toFixed(3): 'N/A'}</td>
    <td className="px-2 py-4 mx-auto items-center text-md text-center">{pickrate !== null ? (pickrate?.valueOf()*100).toFixed(2) + "%": 'N/A' }</td>
  </tr>
);

export default function BrawlerPicker() {
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: 'score', direction: 'desc' });
  const { selectBrawler, updatePredictions, selectBrawlerBan, availableBrawlers, selectedBrawlers, selectedMap, brawlerScores, brawlerPickrates, currentPlayerBrawlers, filterPlayerBrawlers, brawlerBans } = useBrawler();
  const [localAvailableBrawlers, setLocalAvailableBrawlers] = useState<BrawlerPickerProps[]>(availableBrawlers);

  useEffect(() => {
    setLocalAvailableBrawlers(availableBrawlers);
  }, [availableBrawlers]);

  const isMapSelected = selectedMap !== "";

  const handleClick = (brawler: BrawlerPickerProps) => {
    if (!isMapSelected) return;
    const emptySlot = selectedBrawlers.findIndex(slot => slot === null);
    if (emptySlot !== -1) {
      selectBrawler(brawler, emptySlot);
      setFilter("");
    } else {
      alert("All slots are filled. Clear a slot before selecting a new brawler.");
    }
  };

  const handleRightClick = (brawler: BrawlerPickerProps) => {
    if (!isMapSelected) return;
    if (brawlerBans.length <= 6) {
      selectBrawlerBan(brawler);
      setFilter("");
    } else {
      alert("All slots are filled. Clear a slot before banning a new brawler.");
    }
  };

  const filterBrawlers = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!isMapSelected) return;
    setFilter(e.target.value);
  };

  const handleSort = (key: string) => {
    if (!isMapSelected) return;
    setSort(prevSort => ({
      key,
      direction: prevSort.key === key && prevSort.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const filteredAndSortedBrawlers = useMemo(() => {
    const playerBrawlers = currentPlayerBrawlers.map(brawler => brawler.toLowerCase());
    console.log("current player brawlers: ", playerBrawlers)
    const playerBans = brawlerBans.map(brawlerBan => brawlerBan.name.toLowerCase());
    console.log("player bans: ", playerBans)
  
    return localAvailableBrawlers
      .filter(brawler => brawler.name.toLowerCase().includes(filter.toLowerCase()))
      .filter(brawler => {
        if (filterPlayerBrawlers && currentPlayerBrawlers.length > 0) {
          return playerBrawlers.includes(brawler.name.toLowerCase());
        }
        return true;
      })
      .filter(brawler => {
        if (filterPlayerBrawlers && currentPlayerBrawlers.length > 0) {
          return !playerBans.includes(brawler.name.toLowerCase());
        }
        return true;
      })
      .sort((a, b) => {
        if (sort.key === 'score') {
          const scoreA = brawlerScores[a.name.toLowerCase()] ?? -Infinity;
          const scoreB = brawlerScores[b.name.toLowerCase()] ?? -Infinity;
          return sort.direction === 'desc' ? scoreB - scoreA : scoreA - scoreB;
        }
        if (sort.key === 'pickRate') {
          const pickRateA = brawlerPickrates[a.name.toLowerCase()] ?? -Infinity;
          const pickRateB = brawlerPickrates[b.name.toLowerCase()] ?? -Infinity;
          return sort.direction === 'desc' ? pickRateB - pickRateA : pickRateA - pickRateB;
        }
        return 0;
      });
  }, [localAvailableBrawlers, filter, sort, brawlerScores, brawlerPickrates, currentPlayerBrawlers, filterPlayerBrawlers, brawlerBans]);

  return (
    <div className="relative">
      <input
        type="text"
        placeholder="Filter brawlers..."
        value={filter}
        onChange={filterBrawlers}
        className={`w-full p-2 mb-4 border rounded-xl ${!isMapSelected ? 'cursor-not-allowed opacity-50' : ''}`}
        disabled={!isMapSelected}
      />
      <div className={`relative overflow-x-auto h-[500px] shadow-md rounded-xl bg-gray-800 custom-scrollbar ${!isMapSelected ? 'pointer-events-none' : ''}`}>
        <div className="min-w-[250px]">
          <table className="w-full text-sm text-left rtl:text-right text-gray-400">
            <thead className="text-xs uppercase bg-gray-700 text-gray-400">
              <tr>
                <TableHeader title="Brawler" sortable={false} currentSort={sort} onSort={handleSort} />
                <TableHeader title="Score" sortable sortKey="score" currentSort={sort} onSort={handleSort} />
                <TableHeader title="Pick Rate" sortable sortKey="pickRate" currentSort={sort} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {filteredAndSortedBrawlers.map((brawler, index) => (
                <TableRow 
                  key={brawler.name} 
                  brawler={brawler}
                  score={brawlerScores[brawler.name.toLowerCase()] ?? null}
                  pickrate={brawlerPickrates[brawler.name.toLowerCase()] ?? null}
                  onClick={handleClick}
                  onRightClick={handleRightClick}
                  disabled={!isMapSelected}
                  isAboveFold={index < 6}
                />
              ))}
            </tbody>
          </table>
        </div>
        {!isMapSelected && (
        <div className="absolute inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center rounded-xl">
          <span className="text-white font-bold text-2xl">Select Map</span>
        </div>
      )}
      </div>
      
    </div>
  );
}