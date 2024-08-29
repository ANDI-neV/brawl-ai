import React, { useState, useMemo } from 'react';
import BrawlerIcon from "./brawler-icon";
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json";
import { useBrawler } from './brawler-context';

function get_brawlers(): BrawlerPickerProps[] {
  return Object.keys(brawlerJson).map(name => ({
    name,
    score: -1,
    pickrate: -1
  }));
}

interface BrawlerPickerProps {
  name: string;
  score: number;
  pickrate: number;
}

interface TableHeaderProps {
  title: string;
  sortable?: boolean;
}

const TableHeader: React.FC<TableHeaderProps> = ({ title, sortable = false }) => (
  <th scope="col" className="px-6 py-3">
    <div className="flex items-center">
      {title}
      {sortable && (
        <a href="#">
          <svg className="w-3 h-3 ms-1.5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8.574 11.024h6.852a2.075 2.075 0 0 0 1.847-1.086 1.9 1.9 0 0 0-.11-1.986L13.736 2.9a2.122 2.122 0 0 0-3.472 0L6.837 7.952a1.9 1.9 0 0 0-.11 1.986 2.074 2.074 0 0 0 1.847 1.086Zm6.852 1.952H8.574a2.072 2.072 0 0 0-1.847 1.087 1.9 1.9 0 0 0 .11 1.985l3.426 5.05a2.123 2.123 0 0 0 3.472 0l3.427-5.05a1.9 1.9 0 0 0 .11-1.985 2.074 2.074 0 0 0-1.846-1.087Z"/>
          </svg>
        </a>
      )}
    </div>
  </th>
);

interface TableRowProps {
  brawler: BrawlerPickerProps;
  onClick: (brawler: BrawlerPickerProps) => void;
}

const TableRow: React.FC<TableRowProps> = ({ brawler, onClick }) => (
  <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600" onClick={() => onClick(brawler)}>
    <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
      <BrawlerIcon brawler={brawler.name}/>
    </th>
    <td className="px-6 mx-auto items-center py-4">{brawler.score}</td>
    <td className="px-6 py-4">{brawler.pickrate}</td>
  </tr>
);

export default function BrawlerPicker() {
  const [filter, setFilter] = useState("");
  const { selectBrawler, availableBrawlers, selectedBrawlers } = useBrawler();

  const handleClick = (brawler: BrawlerPickerProps) => {
    const emptySlot = selectedBrawlers.findIndex(slot => slot === null);
    if (emptySlot !== -1) {
      selectBrawler(brawler, emptySlot);
    } else {
      alert("All slots are filled. Clear a slot before selecting a new brawler.");
    }
  };

  const filterBrawlers = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilter(e.target.value);
  };

  const filteredBrawlers = useMemo(() => {
    return availableBrawlers.filter(brawler =>
      brawler.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [availableBrawlers, filter]);

  return (
    <>
      <input
          type="text"
          placeholder="Filter brawlers..."
          onChange={filterBrawlers}
          className="w-full p-2 mb-4 border rounded-xl"
        />
      <div className="relative overflow-x-auto h-[500px] shadow-md sm:rounded-lg">
        <div className="min-w-[250px]">
          <table className="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
              <tr>
                <TableHeader title="Brawler" />
                <TableHeader title="Score" sortable />
                <TableHeader title="Pick Rate" sortable />
              </tr>
            </thead>
            <tbody>
              {filteredBrawlers.map((brawler) => (
                <TableRow 
                  key={brawler.name} 
                  brawler={brawler} 
                  onClick={handleClick}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
