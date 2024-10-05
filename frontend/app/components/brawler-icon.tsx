"use client";
import React from 'react';
import Image from "next/image";

interface BrawlerIconProps {
  brawler: string;
}

export default function BrawlerIcon({ brawler }: BrawlerIconProps) {
  const brawler_image_url = `/brawler_images/${brawler}.png`;

  return (
    <div className="md:h-[100px] md:w-[100px] h-[75px] w-[75px] border-4 md:border-8 border-black bg-white bg-opacity-10 p-2 relative rounded-xl overflow-hidden">
      <Image 
        className="rounded-xl object-cover object-left"
        src={brawler_image_url} 
        alt={brawler} 
        fill
        sizes="(max-width: 768px) 75px, 100px"
        style={{
          objectFit: 'cover',
          objectPosition: 'left center'
        }}
      />
      <div className='absolute bottom-0 left-0 right-0 bg-slate-600 bg-opacity-60 text-white text-xs md:text-s rounded-b-xl font-bold text-center p-1'> 
        {brawler}
      </div>
    </div>
  );


}