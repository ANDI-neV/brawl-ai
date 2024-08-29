"use client"
import React from 'react';
import Image from "next/image";
import { useBrawler } from './brawler-context';
import { motion } from 'framer-motion';

interface BrawlerSlotProps {
  team: 'left' | 'right';
  index: number;
  bgColor: string;
}

const first_pick_sequence = [0, 3, 4, 1, 2, 5];
const not_first_pick_sequence = [3, 0, 1, 4, 5, 2];

export default function BrawlerSlot({index, bgColor }: BrawlerSlotProps) {
  const { selectedBrawlers, firstPick, clearSlot } = useBrawler();
  const sequence = firstPick ? first_pick_sequence : not_first_pick_sequence;
  const slotIndex = sequence.indexOf(index);
  const selectedBrawler = selectedBrawlers[slotIndex];
  
  const handleClearSlot = () => {
    if (selectedBrawler) {
      clearSlot(slotIndex);
    }
  };

  if (!selectedBrawler) {
    return <div className="h-[125px] w-[125px] border p-3 rounded-xl" style={{backgroundColor: bgColor}}></div>;
  }

  const brawler_image_url = `/brawler_images/${selectedBrawler.name}.png`;

  return (
    <motion.button 
      className="relative h-[125px] w-[125px] border p-3 rounded-xl overflow-hidden"
      style={{ backgroundColor: bgColor }}
      onClick={handleClearSlot}
      whileHover={{ scale: 1.1, zIndex: 10 }}
      whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
    >
      <Image 
        className="rounded-xl"
        src={brawler_image_url} 
        alt={selectedBrawler.name} 
        width={200} 
        height={200}
        layout="responsive"
      />
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