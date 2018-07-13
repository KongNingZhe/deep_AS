$annotation_path = '.././data/homo_sapiens/trans';
$protein_path = '.././Z_bysj/data/protein_transcript';

open annotation,"<$annotation_path" or die$!;
open OUT,">$protein_path" or die$!;

@chr = (1..22);
push (@chr,'X','Y');

foreach $chri (@chr)
{
	$chrhash{$chri} = 1;
}
$mode = '+';
while($line = <annotation>)
{
	chomp $line;
	G:
	@words=split(" ",$line);
	if((exists($chrhash{$words[0]})) && ($words[1] eq 'protein_coding') && ($words[2] eq 'gene'))
	{
		print OUT "gene\t$mode\n";
		while ($line = <annotation>)
		{
			@words=split(" ",$line);
			if($words[2] eq 'transcript')
			{
				T:
				push(@path ,'@');
				while($line = <annotation>)
				{
					@words = split(" ",$line);
					if($words[2] eq 'exon')
					{
						push(@path,$words[3]);
						push(@path,$words[4]);
					}
					if($words[2] eq 'gene')
					{
						push (@path,'#');
						print OUT "@path\t";
						print OUT "1\t$words[0]\n";
						#删除这个数组
						@path = ();
						$mode = $words[6];
						goto G;
					}
					if($words[2] eq 'transcript')
					{
						push (@path,'#');
						print OUT "@path\t";
						print OUT "1\t$words[0]\n";
						#删除这个数组
						@path = ();
						goto T;
					}
				}
			}
		}
	}
}
