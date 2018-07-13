use strict ;
#use warnings ;


# open in ,"<../data64/transcript" or die$!;
# open out,">../data64/transgraph_zc" or die$!;

sub node {
    my ($value) = @_ ;
	my %hash = ();
    my $self = {
		'child' => /%hash,
        'value' => $value,
    } ;

    return $self ;
}
sub addnode
{
	my ($pre,$beh) = @_;
	print $beh;
	if (exists (${$pre->{child}}{$beh}))
	{
		print "exists\n";
		return  ${$pre->{child}}{$beh};
	}
	else 
	{
		print "new\n";
		print "childs1 add前:";
		print keys %{$pre->{child}};

		%hash = (value =>'ad',child =>'ap');
		$n = \%hash;
		#$m = &node(2);
		
		print $n;
		print $n->{value};

		${$pre->{child}}{$beh} =1;
		
		print $pre;

		#print ${$pre->{child}}{$beh};
		print "\nchilds1 add后:";
		print keys %{$pre->{child}};
		print "\n";
		return ${$pre->{child}} {$beh};
	}

}

sub addpath
{
    my ($point,@seq)=@_;
    my $num=@seq;
	print $num;
	for($i = 1;$i<$num-1;$i++)
	{
		$point = addnode($point,$seq[$i]);
	}
}
sub deletetree
{
	($point) = @_;
	undef $point;
}
sub printtree
{
	my ($point) = @_;
	if($point->{value} eq '#')
	{
		print out "$point->{value}\n";
	}
	else 
	{
		print "$point->{value}\t";
		print out "$point->{value}\t";
		foreach $i (keys %{$pre->{child}})
		{
			printtree(${$pre->{child}}{$i});
		}
	}
}


sub main{
	my $root = &node('@');
	#addpath($root,@array);
	&addnode($root,1);
	&addnode($root,2);
	&addnode($root,1);
#rinttree($root);
	print keys %{$root->{child}};
}
main();
# my $line;
# while($line = <in>)
# {
# 	chomp $line;
# 	my @words=split('\t',$line);
# 	if($line eq 'gene')
# 	{
# 		#print here
# 		&printtree($root);
# 		#clear some hash or data delete the tree
# 		%tranccript = ();
# 		
# 		my $root = node('@');

# 	}
# 	else
# 	{
# 		&addpath($root,@words);
# 	}
# }