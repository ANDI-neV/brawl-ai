"use client";
import React from 'react';
import Image from 'next/image';

interface BrawlerIconProps {
  brawler: string;
}

export default function BrawlerIcon({ brawler }: BrawlerIconProps) {
  const brawler_image_url = `/brawler_images/${brawler}.png`;

  return (
    <div className="md:h-[100px] md:w-[100px] h-[75px] w-[75px] border-2 bg-white bg-opacity-10 p-2 relative rounded-xl">
      <Image className='rounded-xl'
        src={brawler_image_url} 
        alt={brawler} 
        width={200}
        height={200}
        layout='responsive'
      />
      <div className='absolute bottom-0 left-0 right-0 bg-white bg-opacity-20 text-white text-s rounded-b-xl text-center'> 
      {brawler}
      </div>
    </div>
  );
}