%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% quicksort
%   x(jdex,kv): input vector
%   sdex: sorted index
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function sdex = qsort(x,jdex,kv)
L=length(jdex);
sdex=jdex;
sort_r(1,L);

    function sort_r(a,b)
        high=b;
        mid=fix((a+b)/2);
        temp=sdex(mid);
        sdex(mid)=sdex(a);
        sdex(a)=temp;
        pp=a;
        while(pp<high)
            if x(sdex(pp),kv) > x(sdex(pp+1),kv)
                temp=sdex(pp);
                sdex(pp)=sdex(pp+1);
                sdex(pp+1)=temp;
                pp=pp+1;
            else
                temp=sdex(pp+1);
                sdex(pp+1)=sdex(high);
                sdex(high)=temp;
                high=high-1;
            end
        end
        if (pp-1-a > 0)
            sort_r(a,pp-1);
        end
        if (b-pp-1>0)
            sort_r(pp+1,b);
        end
    end
end