import React from 'react';
import { motion } from 'framer-motion';
import Image from "next/image";
import { useBrawler } from './brawler-context';

interface BrawlerSlotProps {
  index: number;
  bgColor: string;
}

const first_pick_sequence = [0, 3, 4, 1, 2, 5];
const not_first_pick_sequence = [3, 0, 1, 4, 5, 2];

function getFirstClearSlot(selectedBrawlers: any[]) {
  return selectedBrawlers.findIndex(brawler => brawler === null);
}

function lightenColor(color: string, amount: number) {
  color = color.replace(/^#/, '');

  let r = parseInt(color.slice(0, 2), 16);
  let g = parseInt(color.slice(2, 4), 16);
  let b = parseInt(color.slice(4, 6), 16);

  r = Math.min(255, Math.round(r + (255 - r) * amount));
  g = Math.min(255, Math.round(g + (255 - g) * amount));
  b = Math.min(255, Math.round(b + (255 - b) * amount));

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

export default function BrawlerSlot({ index, bgColor }: BrawlerSlotProps) {
  const { selectedBrawlers, firstPick, selectedMap, clearSlot, updatePredictions} = useBrawler();
  const sequence = firstPick ? first_pick_sequence : not_first_pick_sequence;
  const slotIndex = sequence.indexOf(index);
  const selectedBrawler = selectedBrawlers[slotIndex];
  
  const firstClearSlotIndex = getFirstClearSlot(selectedBrawlers);
  const isFirstClearSlot = slotIndex === firstClearSlotIndex;

  const handleClearSlot = () => {
    if (selectedBrawler) {
      clearSlot(slotIndex);
      updatePredictions(selectedMap, selectedBrawlers.filter(Boolean).map(b => b!.name), firstPick);
    }
  };

  const lighterBgColor = lightenColor(bgColor, 0.3); // 30% lighter
  const lighterBgColor2 = lightenColor(bgColor, 0.15); // 15% lighter

  if (!selectedBrawler) {
    return (
      <motion.div 
        className={`h-[125px] w-[125px] border-8 p-3 rounded-xl shadow-md ${
          isFirstClearSlot ? '' : 'border-transparent'
        }`}
        style={{
          backgroundColor: bgColor,
          borderColor: isFirstClearSlot ? lighterBgColor : 'transparent',
        }}
        animate={ isFirstClearSlot ? { scale: [1, 1.05, 1], backgroundColor: [lighterBgColor2, bgColor, lighterBgColor2] } : {}}
        transition={isFirstClearSlot ? {
          duration: 2,
          times: [0, 0.5, 1],
          repeat: Infinity,
          ease: "easeInOut",
        } : {}}

      >
        {isFirstClearSlot && (
          <div className="h-full flex items-center justify-center text-white font-bold">
            Next Pick
          </div>
        )}
      </motion.div>
    );
  }

  const brawler_image_url = `/brawler_images/${selectedBrawler.name}.png`;

  return (
    <motion.button 
      className="relative h-[125px] w-[125px] border-8 rounded-2xl overflow-hidden shadow-md"
      style={{ backgroundColor: bgColor, borderColor: lighterBgColor }}
      onClick={handleClearSlot}
      whileHover={{ scale: 1.1, zIndex: 10 }}
      whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
    >
      <div className="relative w-full h-full">
        <Image 
          className="rounded-xl object-cover object-left"
          src={brawler_image_url} 
          alt={selectedBrawler.name} 
          fill
          sizes="125px"
          style={{
            objectFit: 'cover',
            objectPosition: 'left center'
          }}
        />
      </div>
      <motion.div
        className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white font-bold"
        initial={{ opacity: 0 }}
        whileHover={{ opacity: 1 }}
      >
        Remove
      </motion.div>
    </motion.button>
  );

}

