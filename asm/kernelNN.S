
./build/nn:     file format elf64-x86-64


Disassembly of section .init:

00000000004010b0 <_init>:
  4010b0:	48 83 ec 08          	sub    $0x8,%rsp
  4010b4:	48 8b 05 3d 3f 20 00 	mov    0x203f3d(%rip),%rax        # 604ff8 <__gmon_start__>
  4010bb:	48 85 c0             	test   %rax,%rax
  4010be:	74 05                	je     4010c5 <_init+0x15>
  4010c0:	e8 8b 00 00 00       	callq  401150 <__gmon_start__@plt>
  4010c5:	48 83 c4 08          	add    $0x8,%rsp
  4010c9:	c3                   	retq   

Disassembly of section .plt:

00000000004010d0 <.plt>:
  4010d0:	ff 35 32 3f 20 00    	pushq  0x203f32(%rip)        # 605008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4010d6:	ff 25 34 3f 20 00    	jmpq   *0x203f34(%rip)        # 605010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4010dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004010e0 <__cxa_begin_catch@plt>:
  4010e0:	ff 25 32 3f 20 00    	jmpq   *0x203f32(%rip)        # 605018 <__cxa_begin_catch@CXXABI_1.3>
  4010e6:	68 00 00 00 00       	pushq  $0x0
  4010eb:	e9 e0 ff ff ff       	jmpq   4010d0 <.plt>

00000000004010f0 <memcpy@plt>:
  4010f0:	ff 25 2a 3f 20 00    	jmpq   *0x203f2a(%rip)        # 605020 <memcpy@GLIBC_2.14>
  4010f6:	68 01 00 00 00       	pushq  $0x1
  4010fb:	e9 d0 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401100 <_ZN2cv3Mat8copySizeERKS0_@plt>:
  401100:	ff 25 22 3f 20 00    	jmpq   *0x203f22(%rip)        # 605028 <_ZN2cv3Mat8copySizeERKS0_>
  401106:	68 02 00 00 00       	pushq  $0x2
  40110b:	e9 c0 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401110 <_ZSt17__throw_bad_allocv@plt>:
  401110:	ff 25 1a 3f 20 00    	jmpq   *0x203f1a(%rip)        # 605030 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  401116:	68 03 00 00 00       	pushq  $0x3
  40111b:	e9 b0 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401120 <_Znwm@plt>:
  401120:	ff 25 12 3f 20 00    	jmpq   *0x203f12(%rip)        # 605038 <_Znwm@GLIBCXX_3.4>
  401126:	68 04 00 00 00       	pushq  $0x4
  40112b:	e9 a0 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401130 <free@plt>:
  401130:	ff 25 0a 3f 20 00    	jmpq   *0x203f0a(%rip)        # 605040 <free@GLIBC_2.2.5>
  401136:	68 05 00 00 00       	pushq  $0x5
  40113b:	e9 90 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401140 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>:
  401140:	ff 25 02 3f 20 00    	jmpq   *0x203f02(%rip)        # 605048 <_ZN2cv12_OutputArrayC1ERNS_3MatE>
  401146:	68 06 00 00 00       	pushq  $0x6
  40114b:	e9 80 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401150 <__gmon_start__@plt>:
  401150:	ff 25 fa 3e 20 00    	jmpq   *0x203efa(%rip)        # 605050 <__gmon_start__>
  401156:	68 07 00 00 00       	pushq  $0x7
  40115b:	e9 70 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401160 <_ZN2cv3Mat10deallocateEv@plt>:
  401160:	ff 25 f2 3e 20 00    	jmpq   *0x203ef2(%rip)        # 605058 <_ZN2cv3Mat10deallocateEv>
  401166:	68 08 00 00 00       	pushq  $0x8
  40116b:	e9 60 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401170 <_ZdlPv@plt>:
  401170:	ff 25 ea 3e 20 00    	jmpq   *0x203eea(%rip)        # 605060 <_ZdlPv@GLIBCXX_3.4>
  401176:	68 09 00 00 00       	pushq  $0x9
  40117b:	e9 50 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401180 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
  401180:	ff 25 e2 3e 20 00    	jmpq   *0x203ee2(%rip)        # 605068 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
  401186:	68 0a 00 00 00       	pushq  $0xa
  40118b:	e9 40 ff ff ff       	jmpq   4010d0 <.plt>

0000000000401190 <__gxx_personality_v0@plt>:
  401190:	ff 25 da 3e 20 00    	jmpq   *0x203eda(%rip)        # 605070 <__gxx_personality_v0@CXXABI_1.3>
  401196:	68 0b 00 00 00       	pushq  $0xb
  40119b:	e9 30 ff ff ff       	jmpq   4010d0 <.plt>

00000000004011a0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>:
  4011a0:	ff 25 d2 3e 20 00    	jmpq   *0x203ed2(%rip)        # 605078 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE>
  4011a6:	68 0c 00 00 00       	pushq  $0xc
  4011ab:	e9 20 ff ff ff       	jmpq   4010d0 <.plt>

00000000004011b0 <_Unwind_Resume@plt>:
  4011b0:	ff 25 ca 3e 20 00    	jmpq   *0x203eca(%rip)        # 605080 <_Unwind_Resume@GCC_3.0>
  4011b6:	68 0d 00 00 00       	pushq  $0xd
  4011bb:	e9 10 ff ff ff       	jmpq   4010d0 <.plt>

00000000004011c0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4011c0:	ff 25 c2 3e 20 00    	jmpq   *0x203ec2(%rip)        # 605088 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  4011c6:	68 0e 00 00 00       	pushq  $0xe
  4011cb:	e9 00 ff ff ff       	jmpq   4010d0 <.plt>

00000000004011d0 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>:
  4011d0:	ff 25 ba 3e 20 00    	jmpq   *0x203eba(%rip)        # 605090 <_ZN2cv11_InputArrayC1ERKNS_3MatE>
  4011d6:	68 0f 00 00 00       	pushq  $0xf
  4011db:	e9 f0 fe ff ff       	jmpq   4010d0 <.plt>

00000000004011e0 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>:
  4011e0:	ff 25 b2 3e 20 00    	jmpq   *0x203eb2(%rip)        # 605098 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE>
  4011e6:	68 10 00 00 00       	pushq  $0x10
  4011eb:	e9 e0 fe ff ff       	jmpq   4010d0 <.plt>

00000000004011f0 <__cxa_end_catch@plt>:
  4011f0:	ff 25 aa 3e 20 00    	jmpq   *0x203eaa(%rip)        # 6050a0 <__cxa_end_catch@CXXABI_1.3>
  4011f6:	68 11 00 00 00       	pushq  $0x11
  4011fb:	e9 d0 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401200 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>:
  401200:	ff 25 a2 3e 20 00    	jmpq   *0x203ea2(%rip)        # 6050a8 <_ZNSs4_Rep10_M_disposeERKSaIcE@GLIBCXX_3.4>
  401206:	68 12 00 00 00       	pushq  $0x12
  40120b:	e9 c0 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401210 <__cxa_rethrow@plt>:
  401210:	ff 25 9a 3e 20 00    	jmpq   *0x203e9a(%rip)        # 6050b0 <__cxa_rethrow@CXXABI_1.3>
  401216:	68 13 00 00 00       	pushq  $0x13
  40121b:	e9 b0 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401220 <calloc@plt>:
  401220:	ff 25 92 3e 20 00    	jmpq   *0x203e92(%rip)        # 6050b8 <calloc@GLIBC_2.2.5>
  401226:	68 14 00 00 00       	pushq  $0x14
  40122b:	e9 a0 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401230 <__cxa_atexit@plt>:
  401230:	ff 25 8a 3e 20 00    	jmpq   *0x203e8a(%rip)        # 6050c0 <__cxa_atexit@GLIBC_2.2.5>
  401236:	68 15 00 00 00       	pushq  $0x15
  40123b:	e9 90 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401240 <_ZN2cv8fastFreeEPv@plt>:
  401240:	ff 25 82 3e 20 00    	jmpq   *0x203e82(%rip)        # 6050c8 <_ZN2cv8fastFreeEPv>
  401246:	68 16 00 00 00       	pushq  $0x16
  40124b:	e9 80 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401250 <__pthread_key_create@plt>:
  401250:	ff 25 7a 3e 20 00    	jmpq   *0x203e7a(%rip)        # 6050d0 <__pthread_key_create>
  401256:	68 17 00 00 00       	pushq  $0x17
  40125b:	e9 70 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401260 <__libc_start_main@plt>:
  401260:	ff 25 72 3e 20 00    	jmpq   *0x203e72(%rip)        # 6050d8 <__libc_start_main@GLIBC_2.2.5>
  401266:	68 18 00 00 00       	pushq  $0x18
  40126b:	e9 60 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401270 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>:
  401270:	ff 25 6a 3e 20 00    	jmpq   *0x203e6a(%rip)        # 6050e0 <_ZNSs4_Rep10_M_destroyERKSaIcE@GLIBCXX_3.4>
  401276:	68 19 00 00 00       	pushq  $0x19
  40127b:	e9 50 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401280 <_ZNSt8ios_base4InitC1Ev@plt>:
  401280:	ff 25 62 3e 20 00    	jmpq   *0x203e62(%rip)        # 6050e8 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  401286:	68 1a 00 00 00       	pushq  $0x1a
  40128b:	e9 40 fe ff ff       	jmpq   4010d0 <.plt>

0000000000401290 <_ZN2cv5splitERKNS_3MatEPS0_@plt>:
  401290:	ff 25 5a 3e 20 00    	jmpq   *0x203e5a(%rip)        # 6050f0 <_ZN2cv5splitERKNS_3MatEPS0_>
  401296:	68 1b 00 00 00       	pushq  $0x1b
  40129b:	e9 30 fe ff ff       	jmpq   4010d0 <.plt>

00000000004012a0 <_ZN2cv6imreadERKSsi@plt>:
  4012a0:	ff 25 52 3e 20 00    	jmpq   *0x203e52(%rip)        # 6050f8 <_ZN2cv6imreadERKSsi>
  4012a6:	68 1c 00 00 00       	pushq  $0x1c
  4012ab:	e9 20 fe ff ff       	jmpq   4010d0 <.plt>

00000000004012b0 <_ZNSsC1EPKcRKSaIcE@plt>:
  4012b0:	ff 25 4a 3e 20 00    	jmpq   *0x203e4a(%rip)        # 605100 <_ZNSsC1EPKcRKSaIcE@GLIBCXX_3.4>
  4012b6:	68 1d 00 00 00       	pushq  $0x1d
  4012bb:	e9 10 fe ff ff       	jmpq   4010d0 <.plt>

Disassembly of section .text:

00000000004012c0 <main>:
  4012c0:	41 57                	push   %r15
  4012c2:	be 04 00 00 00       	mov    $0x4,%esi
  4012c7:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  4012cc:	41 56                	push   %r14
  4012ce:	41 55                	push   %r13
  4012d0:	41 54                	push   %r12
  4012d2:	55                   	push   %rbp
  4012d3:	53                   	push   %rbx
  4012d4:	48 83 ec 08          	sub    $0x8,%rsp
  4012d8:	e8 43 ff ff ff       	callq  401220 <calloc@plt>
  4012dd:	be 04 00 00 00       	mov    $0x4,%esi
  4012e2:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  4012e7:	49 89 c4             	mov    %rax,%r12
  4012ea:	e8 31 ff ff ff       	callq  401220 <calloc@plt>
  4012ef:	be 04 00 00 00       	mov    $0x4,%esi
  4012f4:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  4012f9:	48 89 c5             	mov    %rax,%rbp
  4012fc:	e8 1f ff ff ff       	callq  401220 <calloc@plt>
  401301:	be 04 00 00 00       	mov    $0x4,%esi
  401306:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  40130b:	48 89 c3             	mov    %rax,%rbx
  40130e:	e8 0d ff ff ff       	callq  401220 <calloc@plt>
  401313:	be 04 00 00 00       	mov    $0x4,%esi
  401318:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  40131d:	49 89 c5             	mov    %rax,%r13
  401320:	e8 fb fe ff ff       	callq  401220 <calloc@plt>
  401325:	be 04 00 00 00       	mov    $0x4,%esi
  40132a:	49 89 c6             	mov    %rax,%r14
  40132d:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  401332:	e8 e9 fe ff ff       	callq  401220 <calloc@plt>
  401337:	4c 89 f6             	mov    %r14,%rsi
  40133a:	4c 89 ef             	mov    %r13,%rdi
  40133d:	48 89 c2             	mov    %rax,%rdx
  401340:	49 89 c7             	mov    %rax,%r15
  401343:	e8 d8 0c 00 00       	callq  402020 <_Z11decodeImagePfS_S_>
  401348:	bf 64 00 00 00       	mov    $0x64,%edi
  40134d:	be 67 66 66 66       	mov    $0x66666667,%esi
  401352:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401358:	45 31 c0             	xor    %r8d,%r8d
  40135b:	31 c9                	xor    %ecx,%ecx
  40135d:	0f 1f 00             	nopl   (%rax)
  401360:	0f 31                	rdtsc  
  401362:	89 c8                	mov    %ecx,%eax
  401364:	41 89 cb             	mov    %ecx,%r11d
  401367:	f7 ee                	imul   %esi
  401369:	89 c8                	mov    %ecx,%eax
  40136b:	c1 f8 1f             	sar    $0x1f,%eax
  40136e:	c4 81 78 28 64 05 00 	vmovaps 0x0(%r13,%r8,1),%xmm4
  401375:	c1 fa 08             	sar    $0x8,%edx
  401378:	c4 81 78 28 14 07    	vmovaps (%r15,%r8,1),%xmm2
  40137e:	29 c2                	sub    %eax,%edx
  401380:	c4 e3 79 04 ec 50    	vpermilps $0x50,%xmm4,%xmm5
  401386:	8d 04 d5 00 00 00 00 	lea    0x0(,%rdx,8),%eax
  40138d:	c4 81 78 28 04 06    	vmovaps (%r14,%r8,1),%xmm0
  401393:	c4 e3 79 04 da 50    	vpermilps $0x50,%xmm2,%xmm3
  401399:	8d 04 50             	lea    (%rax,%rdx,2),%eax
  40139c:	8d 14 92             	lea    (%rdx,%rdx,4),%edx
  40139f:	c4 e3 79 04 c8 50    	vpermilps $0x50,%xmm0,%xmm1
  4013a5:	c1 e2 07             	shl    $0x7,%edx
  4013a8:	c1 e0 08             	shl    $0x8,%eax
  4013ab:	41 29 d3             	sub    %edx,%r11d
  4013ae:	c4 e3 79 04 e4 fa    	vpermilps $0xfa,%xmm4,%xmm4
  4013b4:	42 8d 04 58          	lea    (%rax,%r11,2),%eax
  4013b8:	c4 e3 79 04 d2 fa    	vpermilps $0xfa,%xmm2,%xmm2
  4013be:	c4 e3 79 04 c0 fa    	vpermilps $0xfa,%xmm0,%xmm0
  4013c4:	48 98                	cltq   
  4013c6:	48 c1 e0 02          	shl    $0x2,%rax
  4013ca:	4c 8d 14 03          	lea    (%rbx,%rax,1),%r10
  4013ce:	4c 8d 4c 05 00       	lea    0x0(%rbp,%rax,1),%r9
  4013d3:	4c 01 e0             	add    %r12,%rax
  4013d6:	c5 f8 11 28          	vmovups %xmm5,(%rax)
  4013da:	c5 f8 11 a8 00 14 00 	vmovups %xmm5,0x1400(%rax)
  4013e1:	00 
  4013e2:	c5 f8 11 60 10       	vmovups %xmm4,0x10(%rax)
  4013e7:	c5 f8 11 a0 10 14 00 	vmovups %xmm4,0x1410(%rax)
  4013ee:	00 
  4013ef:	c4 c1 78 11 1a       	vmovups %xmm3,(%r10)
  4013f4:	c4 c1 78 11 9a 00 14 	vmovups %xmm3,0x1400(%r10)
  4013fb:	00 00 
  4013fd:	c4 c1 78 11 52 10    	vmovups %xmm2,0x10(%r10)
  401403:	c4 c1 78 11 92 10 14 	vmovups %xmm2,0x1410(%r10)
  40140a:	00 00 
  40140c:	c4 c1 78 11 09       	vmovups %xmm1,(%r9)
  401411:	c4 c1 78 11 89 00 14 	vmovups %xmm1,0x1400(%r9)
  401418:	00 00 
  40141a:	c4 c1 78 11 41 10    	vmovups %xmm0,0x10(%r9)
  401420:	c4 c1 78 11 81 10 14 	vmovups %xmm0,0x1410(%r9)
  401427:	00 00 
  401429:	0f 31                	rdtsc  
  40142b:	83 c1 04             	add    $0x4,%ecx
  40142e:	49 83 c0 10          	add    $0x10,%r8
  401432:	81 f9 00 b0 04 00    	cmp    $0x4b000,%ecx
  401438:	0f 85 22 ff ff ff    	jne    401360 <main+0xa0>
  40143e:	83 ef 01             	sub    $0x1,%edi
  401441:	0f 85 11 ff ff ff    	jne    401358 <main+0x98>
  401447:	48 89 da             	mov    %rbx,%rdx
  40144a:	b9 a3 3a 40 00       	mov    $0x403aa3,%ecx
  40144f:	48 89 ee             	mov    %rbp,%rsi
  401452:	4c 89 e7             	mov    %r12,%rdi
  401455:	e8 e6 01 00 00       	callq  401640 <_Z11encodeImagePfS_S_Pc>
  40145a:	be c0 3a 40 00       	mov    $0x403ac0,%esi
  40145f:	bf 40 51 60 00       	mov    $0x605140,%edi
  401464:	e8 17 fd ff ff       	callq  401180 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  401469:	4c 89 e7             	mov    %r12,%rdi
  40146c:	e8 bf fc ff ff       	callq  401130 <free@plt>
  401471:	48 89 ef             	mov    %rbp,%rdi
  401474:	e8 b7 fc ff ff       	callq  401130 <free@plt>
  401479:	48 89 df             	mov    %rbx,%rdi
  40147c:	e8 af fc ff ff       	callq  401130 <free@plt>
  401481:	48 83 c4 08          	add    $0x8,%rsp
  401485:	31 c0                	xor    %eax,%eax
  401487:	5b                   	pop    %rbx
  401488:	5d                   	pop    %rbp
  401489:	41 5c                	pop    %r12
  40148b:	41 5d                	pop    %r13
  40148d:	41 5e                	pop    %r14
  40148f:	41 5f                	pop    %r15
  401491:	c3                   	retq   
  401492:	0f 1f 40 00          	nopl   0x0(%rax)
  401496:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40149d:	00 00 00 

00000000004014a0 <_GLOBAL__sub_I__Z11decodeImagePfS_S_>:
  4014a0:	48 83 ec 08          	sub    $0x8,%rsp
  4014a4:	bf 51 52 60 00       	mov    $0x605251,%edi
  4014a9:	e8 d2 fd ff ff       	callq  401280 <_ZNSt8ios_base4InitC1Ev@plt>
  4014ae:	ba 88 3a 40 00       	mov    $0x403a88,%edx
  4014b3:	be 51 52 60 00       	mov    $0x605251,%esi
  4014b8:	bf c0 11 40 00       	mov    $0x4011c0,%edi
  4014bd:	48 83 c4 08          	add    $0x8,%rsp
  4014c1:	e9 6a fd ff ff       	jmpq   401230 <__cxa_atexit@plt>

00000000004014c6 <_start>:
  4014c6:	31 ed                	xor    %ebp,%ebp
  4014c8:	49 89 d1             	mov    %rdx,%r9
  4014cb:	5e                   	pop    %rsi
  4014cc:	48 89 e2             	mov    %rsp,%rdx
  4014cf:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4014d3:	50                   	push   %rax
  4014d4:	54                   	push   %rsp
  4014d5:	49 c7 c0 70 3a 40 00 	mov    $0x403a70,%r8
  4014dc:	48 c7 c1 00 3a 40 00 	mov    $0x403a00,%rcx
  4014e3:	48 c7 c7 c0 12 40 00 	mov    $0x4012c0,%rdi
  4014ea:	e8 71 fd ff ff       	callq  401260 <__libc_start_main@plt>
  4014ef:	f4                   	hlt    

00000000004014f0 <deregister_tm_clones>:
  4014f0:	b8 17 51 60 00       	mov    $0x605117,%eax
  4014f5:	55                   	push   %rbp
  4014f6:	48 2d 10 51 60 00    	sub    $0x605110,%rax
  4014fc:	48 83 f8 0e          	cmp    $0xe,%rax
  401500:	48 89 e5             	mov    %rsp,%rbp
  401503:	77 02                	ja     401507 <deregister_tm_clones+0x17>
  401505:	5d                   	pop    %rbp
  401506:	c3                   	retq   
  401507:	b8 00 00 00 00       	mov    $0x0,%eax
  40150c:	48 85 c0             	test   %rax,%rax
  40150f:	74 f4                	je     401505 <deregister_tm_clones+0x15>
  401511:	5d                   	pop    %rbp
  401512:	bf 10 51 60 00       	mov    $0x605110,%edi
  401517:	ff e0                	jmpq   *%rax
  401519:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401520 <register_tm_clones>:
  401520:	b8 10 51 60 00       	mov    $0x605110,%eax
  401525:	55                   	push   %rbp
  401526:	48 2d 10 51 60 00    	sub    $0x605110,%rax
  40152c:	48 c1 f8 03          	sar    $0x3,%rax
  401530:	48 89 e5             	mov    %rsp,%rbp
  401533:	48 89 c2             	mov    %rax,%rdx
  401536:	48 c1 ea 3f          	shr    $0x3f,%rdx
  40153a:	48 01 d0             	add    %rdx,%rax
  40153d:	48 d1 f8             	sar    %rax
  401540:	75 02                	jne    401544 <register_tm_clones+0x24>
  401542:	5d                   	pop    %rbp
  401543:	c3                   	retq   
  401544:	ba 00 00 00 00       	mov    $0x0,%edx
  401549:	48 85 d2             	test   %rdx,%rdx
  40154c:	74 f4                	je     401542 <register_tm_clones+0x22>
  40154e:	5d                   	pop    %rbp
  40154f:	48 89 c6             	mov    %rax,%rsi
  401552:	bf 10 51 60 00       	mov    $0x605110,%edi
  401557:	ff e2                	jmpq   *%rdx
  401559:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401560 <__do_global_dtors_aux>:
  401560:	80 3d e9 3c 20 00 00 	cmpb   $0x0,0x203ce9(%rip)        # 605250 <completed.6355>
  401567:	75 11                	jne    40157a <__do_global_dtors_aux+0x1a>
  401569:	55                   	push   %rbp
  40156a:	48 89 e5             	mov    %rsp,%rbp
  40156d:	e8 7e ff ff ff       	callq  4014f0 <deregister_tm_clones>
  401572:	5d                   	pop    %rbp
  401573:	c6 05 d6 3c 20 00 01 	movb   $0x1,0x203cd6(%rip)        # 605250 <completed.6355>
  40157a:	f3 c3                	repz retq 
  40157c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401580 <frame_dummy>:
  401580:	48 83 3d 68 37 20 00 	cmpq   $0x0,0x203768(%rip)        # 604cf0 <__JCR_END__>
  401587:	00 
  401588:	74 1e                	je     4015a8 <frame_dummy+0x28>
  40158a:	b8 00 00 00 00       	mov    $0x0,%eax
  40158f:	48 85 c0             	test   %rax,%rax
  401592:	74 14                	je     4015a8 <frame_dummy+0x28>
  401594:	55                   	push   %rbp
  401595:	bf f0 4c 60 00       	mov    $0x604cf0,%edi
  40159a:	48 89 e5             	mov    %rsp,%rbp
  40159d:	ff d0                	callq  *%rax
  40159f:	5d                   	pop    %rbp
  4015a0:	e9 7b ff ff ff       	jmpq   401520 <register_tm_clones>
  4015a5:	0f 1f 00             	nopl   (%rax)
  4015a8:	e9 73 ff ff ff       	jmpq   401520 <register_tm_clones>
  4015ad:	0f 1f 00             	nopl   (%rax)

00000000004015b0 <_Z6kernelPfS_S_S_S_S_i>:
  4015b0:	48 63 44 24 08       	movslq 0x8(%rsp),%rax
  4015b5:	c5 f8 28 12          	vmovaps (%rdx),%xmm2
  4015b9:	c5 f8 28 27          	vmovaps (%rdi),%xmm4
  4015bd:	c4 e3 79 04 da 50    	vpermilps $0x50,%xmm2,%xmm3
  4015c3:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4015ca:	00 
  4015cb:	c5 f8 28 06          	vmovaps (%rsi),%xmm0
  4015cf:	c4 e3 79 04 ec 50    	vpermilps $0x50,%xmm4,%xmm5
  4015d5:	c4 e3 79 04 d2 fa    	vpermilps $0xfa,%xmm2,%xmm2
  4015db:	c4 e3 79 04 c8 50    	vpermilps $0x50,%xmm0,%xmm1
  4015e1:	c4 e3 79 04 e4 fa    	vpermilps $0xfa,%xmm4,%xmm4
  4015e7:	c4 e3 79 04 c0 fa    	vpermilps $0xfa,%xmm0,%xmm0
  4015ed:	c5 f8 29 29          	vmovaps %xmm5,(%rcx)
  4015f1:	c5 f8 29 2c 81       	vmovaps %xmm5,(%rcx,%rax,4)
  4015f6:	c5 f8 29 61 10       	vmovaps %xmm4,0x10(%rcx)
  4015fb:	c5 f8 29 64 11 10    	vmovaps %xmm4,0x10(%rcx,%rdx,1)
  401601:	c4 c1 78 29 19       	vmovaps %xmm3,(%r9)
  401606:	c4 c1 78 29 1c 81    	vmovaps %xmm3,(%r9,%rax,4)
  40160c:	c4 c1 78 29 51 10    	vmovaps %xmm2,0x10(%r9)
  401612:	c4 c1 78 29 54 11 10 	vmovaps %xmm2,0x10(%r9,%rdx,1)
  401619:	c4 c1 78 29 08       	vmovaps %xmm1,(%r8)
  40161e:	c4 c1 78 29 0c 80    	vmovaps %xmm1,(%r8,%rax,4)
  401624:	c4 c1 78 29 40 10    	vmovaps %xmm0,0x10(%r8)
  40162a:	c4 c1 78 29 44 10 10 	vmovaps %xmm0,0x10(%r8,%rdx,1)
  401631:	c3                   	retq   
  401632:	0f 1f 40 00          	nopl   0x0(%rax)
  401636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40163d:	00 00 00 

0000000000401640 <_Z11encodeImagePfS_S_Pc>:
  401640:	55                   	push   %rbp
  401641:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
  401645:	48 89 e5             	mov    %rsp,%rbp
  401648:	41 54                	push   %r12
  40164a:	53                   	push   %rbx
  40164b:	48 89 cb             	mov    %rcx,%rbx
  40164e:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  401652:	48 81 ec 70 02 00 00 	sub    $0x270,%rsp
  401659:	48 8d 84 24 98 00 00 	lea    0x98(%rsp),%rax
  401660:	00 
  401661:	c5 f9 6f 05 87 24 00 	vmovdqa 0x2487(%rip),%xmm0        # 403af0 <__dso_handle+0x68>
  401668:	00 
  401669:	48 89 bc 24 60 01 00 	mov    %rdi,0x160(%rsp)
  401670:	00 
  401671:	48 89 bc 24 70 01 00 	mov    %rdi,0x170(%rsp)
  401678:	00 
  401679:	48 81 c7 00 00 4b 00 	add    $0x4b0000,%rdi
  401680:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  401687:	00 
  401688:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  40168f:	00 
  401690:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  401697:	00 00 
  401699:	c5 f9 7f 8c 24 e0 00 	vmovdqa %xmm1,0xe0(%rsp)
  4016a0:	00 00 
  4016a2:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  4016a9:	00 
  4016aa:	48 8d 84 24 f8 00 00 	lea    0xf8(%rsp),%rax
  4016b1:	00 
  4016b2:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
  4016b9:	00 00 
  4016bb:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  4016c2:	00 00 
  4016c4:	c5 f9 7f 8c 24 40 01 	vmovdqa %xmm1,0x140(%rsp)
  4016cb:	00 00 
  4016cd:	c5 f9 7f 84 24 f0 00 	vmovdqa %xmm0,0xf0(%rsp)
  4016d4:	00 00 
  4016d6:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
  4016dd:	00 
  4016de:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  4016e5:	00 
  4016e6:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  4016ed:	00 00 
  4016ef:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  4016f6:	00 00 00 00 00 
  4016fb:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  401702:	00 00 00 00 00 
  401707:	48 89 84 24 38 01 00 	mov    %rax,0x138(%rsp)
  40170e:	00 
  40170f:	48 8d 84 24 58 01 00 	lea    0x158(%rsp),%rax
  401716:	00 
  401717:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  40171e:	00 00 00 00 00 
  401723:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  40172a:	00 00 00 00 00 
  40172f:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  401736:	00 00 00 00 00 
  40173b:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
  401742:	00 
  401743:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  40174a:	00 
  40174b:	48 c7 84 24 c8 00 00 	movq   $0x0,0xc8(%rsp)
  401752:	00 00 00 00 00 
  401757:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  40175e:	00 00 00 00 00 
  401763:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  40176a:	00 00 00 00 00 
  40176f:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  401776:	00 00 00 00 00 
  40177b:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  401782:	00 00 00 00 00 
  401787:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  40178e:	00 00 00 00 00 
  401793:	48 c7 84 24 28 01 00 	movq   $0x0,0x128(%rsp)
  40179a:	00 00 00 00 00 
  40179f:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  4017a6:	00 00 00 00 00 
  4017ab:	48 c7 84 24 88 01 00 	movq   $0x0,0x188(%rsp)
  4017b2:	00 00 00 00 00 
  4017b7:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  4017be:	00 
  4017bf:	48 8d 84 24 b8 01 00 	lea    0x1b8(%rsp),%rax
  4017c6:	00 
  4017c7:	48 89 94 24 20 02 00 	mov    %rdx,0x220(%rsp)
  4017ce:	00 
  4017cf:	48 89 94 24 30 02 00 	mov    %rdx,0x230(%rsp)
  4017d6:	00 
  4017d7:	48 81 c2 00 00 4b 00 	add    $0x4b0000,%rdx
  4017de:	48 89 bc 24 80 01 00 	mov    %rdi,0x180(%rsp)
  4017e5:	00 
  4017e6:	48 89 84 24 f0 01 00 	mov    %rax,0x1f0(%rsp)
  4017ed:	00 
  4017ee:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  4017f5:	00 
  4017f6:	48 89 bc 24 78 01 00 	mov    %rdi,0x178(%rsp)
  4017fd:	00 
  4017fe:	c5 f9 6f 0d fa 22 00 	vmovdqa 0x22fa(%rip),%xmm1        # 403b00 <__dso_handle+0x78>
  401805:	00 
  401806:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  40180b:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
  401812:	00 
  401813:	48 8d 84 24 18 02 00 	lea    0x218(%rsp),%rax
  40181a:	00 
  40181b:	48 89 b4 24 c0 01 00 	mov    %rsi,0x1c0(%rsp)
  401822:	00 
  401823:	c5 f9 6f 05 e5 22 00 	vmovdqa 0x22e5(%rip),%xmm0        # 403b10 <__dso_handle+0x88>
  40182a:	00 
  40182b:	48 89 b4 24 d0 01 00 	mov    %rsi,0x1d0(%rsp)
  401832:	00 
  401833:	48 89 84 24 50 02 00 	mov    %rax,0x250(%rsp)
  40183a:	00 
  40183b:	48 81 c6 00 00 4b 00 	add    $0x4b0000,%rsi
  401842:	48 8d 84 24 60 02 00 	lea    0x260(%rsp),%rax
  401849:	00 
  40184a:	48 89 94 24 40 02 00 	mov    %rdx,0x240(%rsp)
  401851:	00 
  401852:	48 89 94 24 38 02 00 	mov    %rdx,0x238(%rsp)
  401859:	00 
  40185a:	48 8d 94 24 10 02 00 	lea    0x210(%rsp),%rdx
  401861:	00 
  401862:	48 89 b4 24 e0 01 00 	mov    %rsi,0x1e0(%rsp)
  401869:	00 
  40186a:	48 89 b4 24 d8 01 00 	mov    %rsi,0x1d8(%rsp)
  401871:	00 
  401872:	31 f6                	xor    %esi,%esi
  401874:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  40187b:	00 00 00 00 00 
  401880:	48 c7 84 24 e8 01 00 	movq   $0x0,0x1e8(%rsp)
  401887:	00 00 00 00 00 
  40188c:	c5 f9 7f 8c 24 50 01 	vmovdqa %xmm1,0x150(%rsp)
  401893:	00 00 
  401895:	48 c7 84 24 28 02 00 	movq   $0x0,0x228(%rsp)
  40189c:	00 00 00 00 00 
  4018a1:	48 c7 84 24 48 02 00 	movq   $0x0,0x248(%rsp)
  4018a8:	00 00 00 00 00 
  4018ad:	c5 f9 7f 84 24 a0 01 	vmovdqa %xmm0,0x1a0(%rsp)
  4018b4:	00 00 
  4018b6:	c5 f9 7f 8c 24 b0 01 	vmovdqa %xmm1,0x1b0(%rsp)
  4018bd:	00 00 
  4018bf:	c5 f9 7f 84 24 00 02 	vmovdqa %xmm0,0x200(%rsp)
  4018c6:	00 00 
  4018c8:	c5 f9 7f 8c 24 10 02 	vmovdqa %xmm1,0x210(%rsp)
  4018cf:	00 00 
  4018d1:	c5 f9 7f 84 24 60 02 	vmovdqa %xmm0,0x260(%rsp)
  4018d8:	00 00 
  4018da:	48 89 84 24 58 02 00 	mov    %rax,0x258(%rsp)
  4018e1:	00 
  4018e2:	e8 59 11 00 00       	callq  402a40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  4018e7:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4018ec:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  4018f1:	48 39 c7             	cmp    %rax,%rdi
  4018f4:	0f 84 f7 05 00 00    	je     401ef1 <_Z11encodeImagePfS_S_Pc+0x8b1>
  4018fa:	48 85 ff             	test   %rdi,%rdi
  4018fd:	0f 84 73 05 00 00    	je     401e76 <_Z11encodeImagePfS_S_Pc+0x836>
  401903:	c5 f9 6f 84 24 b0 01 	vmovdqa 0x1b0(%rsp),%xmm0
  40190a:	00 00 
  40190c:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  401910:	48 8b 94 24 d0 01 00 	mov    0x1d0(%rsp),%rdx
  401917:	00 
  401918:	48 8b 84 24 c0 01 00 	mov    0x1c0(%rsp),%rax
  40191f:	00 
  401920:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  401924:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  401928:	48 8b 94 24 d8 01 00 	mov    0x1d8(%rsp),%rdx
  40192f:	00 
  401930:	48 89 47 10          	mov    %rax,0x10(%rdi)
  401934:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  40193b:	00 
  40193c:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  401941:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  401945:	48 8b 94 24 e0 01 00 	mov    0x1e0(%rsp),%rdx
  40194c:	00 
  40194d:	48 85 c0             	test   %rax,%rax
  401950:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  401954:	48 8b 94 24 e8 01 00 	mov    0x1e8(%rsp),%rdx
  40195b:	00 
  40195c:	48 89 47 18          	mov    %rax,0x18(%rdi)
  401960:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  401964:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  401968:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  40196c:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  401970:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  401974:	74 04                	je     40197a <_Z11encodeImagePfS_S_Pc+0x33a>
  401976:	f0 83 00 01          	lock addl $0x1,(%rax)
  40197a:	83 bc 24 b4 01 00 00 	cmpl   $0x2,0x1b4(%rsp)
  401981:	02 
  401982:	0f 8f 88 04 00 00    	jg     401e10 <_Z11encodeImagePfS_S_Pc+0x7d0>
  401988:	48 8b 94 24 f8 01 00 	mov    0x1f8(%rsp),%rdx
  40198f:	00 
  401990:	48 8b 47 48          	mov    0x48(%rdi),%rax
  401994:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  401999:	48 8b 12             	mov    (%rdx),%rdx
  40199c:	48 89 10             	mov    %rdx,(%rax)
  40199f:	48 8b 94 24 f8 01 00 	mov    0x1f8(%rsp),%rdx
  4019a6:	00 
  4019a7:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4019ab:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4019b0:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  4019b4:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4019b8:	48 83 c7 60          	add    $0x60,%rdi
  4019bc:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  4019c1:	48 39 fe             	cmp    %rdi,%rsi
  4019c4:	0f 84 10 05 00 00    	je     401eda <_Z11encodeImagePfS_S_Pc+0x89a>
  4019ca:	48 85 ff             	test   %rdi,%rdi
  4019cd:	0f 84 b0 00 00 00    	je     401a83 <_Z11encodeImagePfS_S_Pc+0x443>
  4019d3:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  4019da:	00 00 
  4019dc:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  4019e0:	48 8b 94 24 70 01 00 	mov    0x170(%rsp),%rdx
  4019e7:	00 
  4019e8:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  4019ef:	00 
  4019f0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4019f4:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  4019f8:	48 8b 94 24 78 01 00 	mov    0x178(%rsp),%rdx
  4019ff:	00 
  401a00:	48 89 47 10          	mov    %rax,0x10(%rdi)
  401a04:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  401a0b:	00 
  401a0c:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  401a11:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  401a15:	48 8b 94 24 80 01 00 	mov    0x180(%rsp),%rdx
  401a1c:	00 
  401a1d:	48 85 c0             	test   %rax,%rax
  401a20:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  401a24:	48 8b 94 24 88 01 00 	mov    0x188(%rsp),%rdx
  401a2b:	00 
  401a2c:	48 89 47 18          	mov    %rax,0x18(%rdi)
  401a30:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  401a34:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  401a38:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  401a3c:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  401a40:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  401a44:	74 04                	je     401a4a <_Z11encodeImagePfS_S_Pc+0x40a>
  401a46:	f0 83 00 01          	lock addl $0x1,(%rax)
  401a4a:	83 bc 24 54 01 00 00 	cmpl   $0x2,0x154(%rsp)
  401a51:	02 
  401a52:	0f 8f 00 04 00 00    	jg     401e58 <_Z11encodeImagePfS_S_Pc+0x818>
  401a58:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  401a5f:	00 
  401a60:	48 8b 47 48          	mov    0x48(%rdi),%rax
  401a64:	48 8b 12             	mov    (%rdx),%rdx
  401a67:	48 89 10             	mov    %rdx,(%rax)
  401a6a:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  401a71:	00 
  401a72:	48 8b 47 48          	mov    0x48(%rdi),%rax
  401a76:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  401a7b:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  401a7f:	48 89 50 08          	mov    %rdx,0x8(%rax)
  401a83:	48 83 c7 60          	add    $0x60,%rdi
  401a87:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  401a8c:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  401a93:	00 
  401a94:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  401a99:	e8 a2 f6 ff ff       	callq  401140 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>
  401a9e:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  401aa3:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  401aa8:	e8 33 f7 ff ff       	callq  4011e0 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>
  401aad:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  401ab4:	00 
  401ab5:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  401aba:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  401ac1:	00 00 
  401ac3:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  401aca:	00 00 
  401acc:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  401ad3:	00 00 
  401ad5:	e8 f6 f6 ff ff       	callq  4011d0 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>
  401ada:	48 8d 54 24 0e       	lea    0xe(%rsp),%rdx
  401adf:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  401ae4:	48 89 de             	mov    %rbx,%rsi
  401ae7:	e8 c4 f7 ff ff       	callq  4012b0 <_ZNSsC1EPKcRKSaIcE@plt>
  401aec:	48 8d 54 24 50       	lea    0x50(%rsp),%rdx
  401af1:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  401af6:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  401afb:	e8 a0 f6 ff ff       	callq  4011a0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>
  401b00:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  401b05:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  401b09:	48 81 ff 20 51 60 00 	cmp    $0x605120,%rdi
  401b10:	0f 85 ff 03 00 00    	jne    401f15 <_Z11encodeImagePfS_S_Pc+0x8d5>
  401b16:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  401b1b:	48 85 ff             	test   %rdi,%rdi
  401b1e:	74 05                	je     401b25 <_Z11encodeImagePfS_S_Pc+0x4e5>
  401b20:	e8 4b f6 ff ff       	callq  401170 <_ZdlPv@plt>
  401b25:	48 8b 84 24 28 02 00 	mov    0x228(%rsp),%rax
  401b2c:	00 
  401b2d:	48 85 c0             	test   %rax,%rax
  401b30:	74 0a                	je     401b3c <_Z11encodeImagePfS_S_Pc+0x4fc>
  401b32:	f0 83 28 01          	lock subl $0x1,(%rax)
  401b36:	0f 84 44 03 00 00    	je     401e80 <_Z11encodeImagePfS_S_Pc+0x840>
  401b3c:	48 8b 84 24 50 02 00 	mov    0x250(%rsp),%rax
  401b43:	00 
  401b44:	48 8b bc 24 58 02 00 	mov    0x258(%rsp),%rdi
  401b4b:	00 
  401b4c:	48 c7 84 24 40 02 00 	movq   $0x0,0x240(%rsp)
  401b53:	00 00 00 00 00 
  401b58:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  401b5f:	00 00 00 00 00 
  401b64:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  401b6b:	00 00 00 00 00 
  401b70:	48 c7 84 24 20 02 00 	movq   $0x0,0x220(%rsp)
  401b77:	00 00 00 00 00 
  401b7c:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401b82:	48 8d 84 24 60 02 00 	lea    0x260(%rsp),%rax
  401b89:	00 
  401b8a:	48 c7 84 24 28 02 00 	movq   $0x0,0x228(%rsp)
  401b91:	00 00 00 00 00 
  401b96:	48 39 c7             	cmp    %rax,%rdi
  401b99:	74 05                	je     401ba0 <_Z11encodeImagePfS_S_Pc+0x560>
  401b9b:	e8 a0 f6 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401ba0:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  401ba7:	00 
  401ba8:	48 85 c0             	test   %rax,%rax
  401bab:	74 0a                	je     401bb7 <_Z11encodeImagePfS_S_Pc+0x577>
  401bad:	f0 83 28 01          	lock subl $0x1,(%rax)
  401bb1:	0f 84 11 03 00 00    	je     401ec8 <_Z11encodeImagePfS_S_Pc+0x888>
  401bb7:	48 8b 84 24 f0 01 00 	mov    0x1f0(%rsp),%rax
  401bbe:	00 
  401bbf:	48 8b bc 24 f8 01 00 	mov    0x1f8(%rsp),%rdi
  401bc6:	00 
  401bc7:	48 c7 84 24 e0 01 00 	movq   $0x0,0x1e0(%rsp)
  401bce:	00 00 00 00 00 
  401bd3:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  401bda:	00 00 00 00 00 
  401bdf:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  401be6:	00 00 00 00 00 
  401beb:	48 c7 84 24 c0 01 00 	movq   $0x0,0x1c0(%rsp)
  401bf2:	00 00 00 00 00 
  401bf7:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401bfd:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  401c04:	00 
  401c05:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  401c0c:	00 00 00 00 00 
  401c11:	48 39 c7             	cmp    %rax,%rdi
  401c14:	74 05                	je     401c1b <_Z11encodeImagePfS_S_Pc+0x5db>
  401c16:	e8 25 f6 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401c1b:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  401c22:	00 
  401c23:	48 85 c0             	test   %rax,%rax
  401c26:	74 0a                	je     401c32 <_Z11encodeImagePfS_S_Pc+0x5f2>
  401c28:	f0 83 28 01          	lock subl $0x1,(%rax)
  401c2c:	0f 84 84 02 00 00    	je     401eb6 <_Z11encodeImagePfS_S_Pc+0x876>
  401c32:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
  401c39:	00 
  401c3a:	48 8b bc 24 98 01 00 	mov    0x198(%rsp),%rdi
  401c41:	00 
  401c42:	48 c7 84 24 80 01 00 	movq   $0x0,0x180(%rsp)
  401c49:	00 00 00 00 00 
  401c4e:	48 c7 84 24 78 01 00 	movq   $0x0,0x178(%rsp)
  401c55:	00 00 00 00 00 
  401c5a:	48 c7 84 24 70 01 00 	movq   $0x0,0x170(%rsp)
  401c61:	00 00 00 00 00 
  401c66:	48 c7 84 24 60 01 00 	movq   $0x0,0x160(%rsp)
  401c6d:	00 00 00 00 00 
  401c72:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401c78:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  401c7f:	00 
  401c80:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  401c87:	00 00 00 00 00 
  401c8c:	48 39 c7             	cmp    %rax,%rdi
  401c8f:	74 05                	je     401c96 <_Z11encodeImagePfS_S_Pc+0x656>
  401c91:	e8 aa f5 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401c96:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  401c9d:	00 
  401c9e:	48 85 c0             	test   %rax,%rax
  401ca1:	74 0a                	je     401cad <_Z11encodeImagePfS_S_Pc+0x66d>
  401ca3:	f0 83 28 01          	lock subl $0x1,(%rax)
  401ca7:	0f 84 f7 01 00 00    	je     401ea4 <_Z11encodeImagePfS_S_Pc+0x864>
  401cad:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  401cb4:	00 
  401cb5:	48 8b bc 24 38 01 00 	mov    0x138(%rsp),%rdi
  401cbc:	00 
  401cbd:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  401cc4:	00 00 00 00 00 
  401cc9:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  401cd0:	00 00 00 00 00 
  401cd5:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  401cdc:	00 00 00 00 00 
  401ce1:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  401ce8:	00 00 00 00 00 
  401ced:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401cf3:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  401cfa:	00 
  401cfb:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  401d02:	00 00 00 00 00 
  401d07:	48 39 c7             	cmp    %rax,%rdi
  401d0a:	74 05                	je     401d11 <_Z11encodeImagePfS_S_Pc+0x6d1>
  401d0c:	e8 2f f5 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401d11:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  401d18:	00 
  401d19:	48 85 c0             	test   %rax,%rax
  401d1c:	74 0a                	je     401d28 <_Z11encodeImagePfS_S_Pc+0x6e8>
  401d1e:	f0 83 28 01          	lock subl $0x1,(%rax)
  401d22:	0f 84 6a 01 00 00    	je     401e92 <_Z11encodeImagePfS_S_Pc+0x852>
  401d28:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
  401d2f:	00 
  401d30:	48 8b bc 24 d8 00 00 	mov    0xd8(%rsp),%rdi
  401d37:	00 
  401d38:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  401d3f:	00 00 00 00 00 
  401d44:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  401d4b:	00 00 00 00 00 
  401d50:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  401d57:	00 00 00 00 00 
  401d5c:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  401d63:	00 00 00 00 00 
  401d68:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401d6e:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  401d75:	00 
  401d76:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  401d7d:	00 00 00 00 00 
  401d82:	48 39 c7             	cmp    %rax,%rdi
  401d85:	74 05                	je     401d8c <_Z11encodeImagePfS_S_Pc+0x74c>
  401d87:	e8 b4 f4 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401d8c:	4c 8b 64 24 38       	mov    0x38(%rsp),%r12
  401d91:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  401d96:	49 39 dc             	cmp    %rbx,%r12
  401d99:	4c 89 e7             	mov    %r12,%rdi
  401d9c:	75 57                	jne    401df5 <_Z11encodeImagePfS_S_Pc+0x7b5>
  401d9e:	e9 9a 00 00 00       	jmpq   401e3d <_Z11encodeImagePfS_S_Pc+0x7fd>
  401da3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401da8:	48 8b 43 40          	mov    0x40(%rbx),%rax
  401dac:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  401db3:	00 
  401db4:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  401dbb:	00 
  401dbc:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  401dc3:	00 
  401dc4:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  401dcb:	00 
  401dcc:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  401dd2:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  401dd6:	48 8d 43 50          	lea    0x50(%rbx),%rax
  401dda:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  401de1:	00 
  401de2:	48 39 c7             	cmp    %rax,%rdi
  401de5:	74 05                	je     401dec <_Z11encodeImagePfS_S_Pc+0x7ac>
  401de7:	e8 54 f4 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  401dec:	48 83 c3 60          	add    $0x60,%rbx
  401df0:	49 39 dc             	cmp    %rbx,%r12
  401df3:	74 43                	je     401e38 <_Z11encodeImagePfS_S_Pc+0x7f8>
  401df5:	48 8b 43 18          	mov    0x18(%rbx),%rax
  401df9:	48 85 c0             	test   %rax,%rax
  401dfc:	74 aa                	je     401da8 <_Z11encodeImagePfS_S_Pc+0x768>
  401dfe:	f0 83 28 01          	lock subl $0x1,(%rax)
  401e02:	75 a4                	jne    401da8 <_Z11encodeImagePfS_S_Pc+0x768>
  401e04:	48 89 df             	mov    %rbx,%rdi
  401e07:	e8 54 f3 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401e0c:	eb 9a                	jmp    401da8 <_Z11encodeImagePfS_S_Pc+0x768>
  401e0e:	66 90                	xchg   %ax,%ax
  401e10:	48 8d b4 24 b0 01 00 	lea    0x1b0(%rsp),%rsi
  401e17:	00 
  401e18:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  401e1f:	e8 dc f2 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  401e24:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  401e29:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  401e2e:	e9 85 fb ff ff       	jmpq   4019b8 <_Z11encodeImagePfS_S_Pc+0x378>
  401e33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401e38:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  401e3d:	48 85 ff             	test   %rdi,%rdi
  401e40:	74 05                	je     401e47 <_Z11encodeImagePfS_S_Pc+0x807>
  401e42:	e8 29 f3 ff ff       	callq  401170 <_ZdlPv@plt>
  401e47:	48 8d 65 f0          	lea    -0x10(%rbp),%rsp
  401e4b:	31 c0                	xor    %eax,%eax
  401e4d:	5b                   	pop    %rbx
  401e4e:	41 5c                	pop    %r12
  401e50:	5d                   	pop    %rbp
  401e51:	c3                   	retq   
  401e52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401e58:	48 8d b4 24 50 01 00 	lea    0x150(%rsp),%rsi
  401e5f:	00 
  401e60:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  401e67:	e8 94 f2 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  401e6c:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  401e71:	e9 0d fc ff ff       	jmpq   401a83 <_Z11encodeImagePfS_S_Pc+0x443>
  401e76:	48 89 c6             	mov    %rax,%rsi
  401e79:	31 ff                	xor    %edi,%edi
  401e7b:	e9 38 fb ff ff       	jmpq   4019b8 <_Z11encodeImagePfS_S_Pc+0x378>
  401e80:	48 8d bc 24 10 02 00 	lea    0x210(%rsp),%rdi
  401e87:	00 
  401e88:	e8 d3 f2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401e8d:	e9 aa fc ff ff       	jmpq   401b3c <_Z11encodeImagePfS_S_Pc+0x4fc>
  401e92:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  401e99:	00 
  401e9a:	e8 c1 f2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401e9f:	e9 84 fe ff ff       	jmpq   401d28 <_Z11encodeImagePfS_S_Pc+0x6e8>
  401ea4:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  401eab:	00 
  401eac:	e8 af f2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401eb1:	e9 f7 fd ff ff       	jmpq   401cad <_Z11encodeImagePfS_S_Pc+0x66d>
  401eb6:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  401ebd:	00 
  401ebe:	e8 9d f2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401ec3:	e9 6a fd ff ff       	jmpq   401c32 <_Z11encodeImagePfS_S_Pc+0x5f2>
  401ec8:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  401ecf:	00 
  401ed0:	e8 8b f2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  401ed5:	e9 dd fc ff ff       	jmpq   401bb7 <_Z11encodeImagePfS_S_Pc+0x577>
  401eda:	48 8d 94 24 50 01 00 	lea    0x150(%rsp),%rdx
  401ee1:	00 
  401ee2:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  401ee7:	e8 54 0b 00 00       	callq  402a40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  401eec:	e9 9b fb ff ff       	jmpq   401a8c <_Z11encodeImagePfS_S_Pc+0x44c>
  401ef1:	48 89 fe             	mov    %rdi,%rsi
  401ef4:	48 8d 94 24 b0 01 00 	lea    0x1b0(%rsp),%rdx
  401efb:	00 
  401efc:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  401f01:	e8 3a 0b 00 00       	callq  402a40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  401f06:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  401f0b:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  401f10:	e9 ac fa ff ff       	jmpq   4019c1 <_Z11encodeImagePfS_S_Pc+0x381>
  401f15:	b9 50 12 40 00       	mov    $0x401250,%ecx
  401f1a:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  401f1e:	48 85 c9             	test   %rcx,%rcx
  401f21:	74 52                	je     401f75 <_Z11encodeImagePfS_S_Pc+0x935>
  401f23:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401f28:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  401f2c:	85 c0                	test   %eax,%eax
  401f2e:	0f 8f e2 fb ff ff    	jg     401b16 <_Z11encodeImagePfS_S_Pc+0x4d6>
  401f34:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  401f39:	e8 32 f3 ff ff       	callq  401270 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  401f3e:	e9 d3 fb ff ff       	jmpq   401b16 <_Z11encodeImagePfS_S_Pc+0x4d6>
  401f43:	48 89 c3             	mov    %rax,%rbx
  401f46:	c5 f8 77             	vzeroupper 
  401f49:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  401f50:	00 
  401f51:	e8 8a 09 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  401f56:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  401f5d:	00 
  401f5e:	e8 7d 09 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  401f63:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  401f68:	e8 f3 09 00 00       	callq  402960 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  401f6d:	48 89 df             	mov    %rbx,%rdi
  401f70:	e8 3b f2 ff ff       	callq  4011b0 <_Unwind_Resume@plt>
  401f75:	8b 50 f8             	mov    -0x8(%rax),%edx
  401f78:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  401f7b:	89 48 f8             	mov    %ecx,-0x8(%rax)
  401f7e:	89 d0                	mov    %edx,%eax
  401f80:	eb aa                	jmp    401f2c <_Z11encodeImagePfS_S_Pc+0x8ec>
  401f82:	48 89 c3             	mov    %rax,%rbx
  401f85:	c5 f8 77             	vzeroupper 
  401f88:	eb cc                	jmp    401f56 <_Z11encodeImagePfS_S_Pc+0x916>
  401f8a:	48 89 c3             	mov    %rax,%rbx
  401f8d:	c5 f8 77             	vzeroupper 
  401f90:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  401f97:	00 
  401f98:	e8 43 09 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  401f9d:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  401fa4:	00 
  401fa5:	e8 36 09 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  401faa:	eb 9d                	jmp    401f49 <_Z11encodeImagePfS_S_Pc+0x909>
  401fac:	48 89 c3             	mov    %rax,%rbx
  401faf:	c5 f8 77             	vzeroupper 
  401fb2:	eb e9                	jmp    401f9d <_Z11encodeImagePfS_S_Pc+0x95d>
  401fb4:	48 89 c3             	mov    %rax,%rbx
  401fb7:	c5 f8 77             	vzeroupper 
  401fba:	eb a7                	jmp    401f63 <_Z11encodeImagePfS_S_Pc+0x923>
  401fbc:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  401fc1:	48 89 c3             	mov    %rax,%rbx
  401fc4:	48 85 ff             	test   %rdi,%rdi
  401fc7:	74 1f                	je     401fe8 <_Z11encodeImagePfS_S_Pc+0x9a8>
  401fc9:	c5 f8 77             	vzeroupper 
  401fcc:	e8 9f f1 ff ff       	callq  401170 <_ZdlPv@plt>
  401fd1:	eb 9a                	jmp    401f6d <_Z11encodeImagePfS_S_Pc+0x92d>
  401fd3:	48 89 c3             	mov    %rax,%rbx
  401fd6:	c5 f8 77             	vzeroupper 
  401fd9:	48 8d bc 24 10 02 00 	lea    0x210(%rsp),%rdi
  401fe0:	00 
  401fe1:	e8 fa 08 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  401fe6:	eb a8                	jmp    401f90 <_Z11encodeImagePfS_S_Pc+0x950>
  401fe8:	c5 f8 77             	vzeroupper 
  401feb:	eb 80                	jmp    401f6d <_Z11encodeImagePfS_S_Pc+0x92d>
  401fed:	48 89 c3             	mov    %rax,%rbx
  401ff0:	c5 f8 77             	vzeroupper 
  401ff3:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  401ff8:	48 85 ff             	test   %rdi,%rdi
  401ffb:	74 dc                	je     401fd9 <_Z11encodeImagePfS_S_Pc+0x999>
  401ffd:	e8 6e f1 ff ff       	callq  401170 <_ZdlPv@plt>
  402002:	eb d5                	jmp    401fd9 <_Z11encodeImagePfS_S_Pc+0x999>
  402004:	48 89 c3             	mov    %rax,%rbx
  402007:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40200c:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  402011:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402015:	c5 f8 77             	vzeroupper 
  402018:	e8 e3 f1 ff ff       	callq  401200 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  40201d:	eb d4                	jmp    401ff3 <_Z11encodeImagePfS_S_Pc+0x9b3>
  40201f:	90                   	nop

0000000000402020 <_Z11decodeImagePfS_S_>:
  402020:	55                   	push   %rbp
  402021:	48 89 e5             	mov    %rsp,%rbp
  402024:	41 57                	push   %r15
  402026:	41 56                	push   %r14
  402028:	41 55                	push   %r13
  40202a:	49 89 d5             	mov    %rdx,%r13
  40202d:	41 54                	push   %r12
  40202f:	49 89 f4             	mov    %rsi,%r12
  402032:	be 90 3a 40 00       	mov    $0x403a90,%esi
  402037:	53                   	push   %rbx
  402038:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40203c:	48 81 ec a0 02 00 00 	sub    $0x2a0,%rsp
  402043:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
  402048:	48 8d 9c 24 80 01 00 	lea    0x180(%rsp),%rbx
  40204f:	00 
  402050:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  402055:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  40205c:	00 00 00 00 00 
  402061:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  402068:	00 00 00 00 00 
  40206d:	48 8d 54 24 1e       	lea    0x1e(%rsp),%rdx
  402072:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  402079:	00 
  40207a:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  402081:	00 
  402082:	c7 44 24 60 00 00 ff 	movl   $0x42ff0000,0x60(%rsp)
  402089:	42 
  40208a:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
  402091:	00 
  402092:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
  402099:	00 
  40209a:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40209f:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  4020a6:	00 
  4020a7:	48 8d 84 24 c8 00 00 	lea    0xc8(%rsp),%rax
  4020ae:	00 
  4020af:	c7 44 24 64 00 00 00 	movl   $0x0,0x64(%rsp)
  4020b6:	00 
  4020b7:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  4020be:	00 00 00 00 00 
  4020c3:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  4020ca:	00 00 00 00 00 
  4020cf:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  4020d6:	00 
  4020d7:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  4020de:	00 
  4020df:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  4020e6:	00 00 00 00 00 
  4020eb:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  4020f2:	00 00 
  4020f4:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  4020fb:	00 00 
  4020fd:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  402104:	00 
  402105:	48 8d 43 08          	lea    0x8(%rbx),%rax
  402109:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
  402110:	00 00 00 00 00 
  402115:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  40211c:	00 00 00 00 00 
  402121:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  402128:	00 00 00 00 00 
  40212d:	c7 84 24 c0 00 00 00 	movl   $0x42ff0000,0xc0(%rsp)
  402134:	00 00 ff 42 
  402138:	c7 84 24 cc 00 00 00 	movl   $0x0,0xcc(%rsp)
  40213f:	00 00 00 00 
  402143:	c7 84 24 c8 00 00 00 	movl   $0x0,0xc8(%rsp)
  40214a:	00 00 00 00 
  40214e:	c7 84 24 c4 00 00 00 	movl   $0x0,0xc4(%rsp)
  402155:	00 00 00 00 
  402159:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402160:	00 00 00 00 00 
  402165:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  40216c:	00 00 00 00 00 
  402171:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  402178:	00 00 00 00 00 
  40217d:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  402184:	00 00 00 00 00 
  402189:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
  402190:	00 
  402191:	48 8d 43 50          	lea    0x50(%rbx),%rax
  402195:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  40219c:	00 00 00 00 00 
  4021a1:	48 c7 84 24 f8 00 00 	movq   $0x0,0xf8(%rsp)
  4021a8:	00 00 00 00 00 
  4021ad:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  4021b4:	00 00 00 00 00 
  4021b9:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
  4021c0:	00 
  4021c1:	48 8d 43 68          	lea    0x68(%rbx),%rax
  4021c5:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  4021cc:	00 00 00 00 00 
  4021d1:	c7 84 24 80 01 00 00 	movl   $0x42ff0000,0x180(%rsp)
  4021d8:	00 00 ff 42 
  4021dc:	c7 84 24 8c 01 00 00 	movl   $0x0,0x18c(%rsp)
  4021e3:	00 00 00 00 
  4021e7:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
  4021ee:	00 
  4021ef:	48 8d 83 b0 00 00 00 	lea    0xb0(%rbx),%rax
  4021f6:	c7 84 24 88 01 00 00 	movl   $0x0,0x188(%rsp)
  4021fd:	00 00 00 00 
  402201:	c7 84 24 84 01 00 00 	movl   $0x0,0x184(%rsp)
  402208:	00 00 00 00 
  40220c:	48 c7 84 24 b0 01 00 	movq   $0x0,0x1b0(%rsp)
  402213:	00 00 00 00 00 
  402218:	48 89 84 24 28 02 00 	mov    %rax,0x228(%rsp)
  40221f:	00 
  402220:	48 8d 83 c8 00 00 00 	lea    0xc8(%rbx),%rax
  402227:	48 c7 84 24 a8 01 00 	movq   $0x0,0x1a8(%rsp)
  40222e:	00 00 00 00 00 
  402233:	48 c7 84 24 a0 01 00 	movq   $0x0,0x1a0(%rsp)
  40223a:	00 00 00 00 00 
  40223f:	48 c7 84 24 90 01 00 	movq   $0x0,0x190(%rsp)
  402246:	00 00 00 00 00 
  40224b:	48 89 84 24 80 02 00 	mov    %rax,0x280(%rsp)
  402252:	00 
  402253:	48 8d 83 10 01 00 00 	lea    0x110(%rbx),%rax
  40225a:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  402261:	00 00 00 00 00 
  402266:	48 c7 84 24 b8 01 00 	movq   $0x0,0x1b8(%rsp)
  40226d:	00 00 00 00 00 
  402272:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  402279:	00 00 00 00 00 
  40227e:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  402285:	00 00 00 00 00 
  40228a:	c7 84 24 e0 01 00 00 	movl   $0x42ff0000,0x1e0(%rsp)
  402291:	00 00 ff 42 
  402295:	c7 84 24 ec 01 00 00 	movl   $0x0,0x1ec(%rsp)
  40229c:	00 00 00 00 
  4022a0:	c7 84 24 e8 01 00 00 	movl   $0x0,0x1e8(%rsp)
  4022a7:	00 00 00 00 
  4022ab:	c7 84 24 e4 01 00 00 	movl   $0x0,0x1e4(%rsp)
  4022b2:	00 00 00 00 
  4022b6:	48 c7 84 24 10 02 00 	movq   $0x0,0x210(%rsp)
  4022bd:	00 00 00 00 00 
  4022c2:	48 c7 84 24 08 02 00 	movq   $0x0,0x208(%rsp)
  4022c9:	00 00 00 00 00 
  4022ce:	48 c7 84 24 00 02 00 	movq   $0x0,0x200(%rsp)
  4022d5:	00 00 00 00 00 
  4022da:	48 c7 84 24 f0 01 00 	movq   $0x0,0x1f0(%rsp)
  4022e1:	00 00 00 00 00 
  4022e6:	48 c7 84 24 f8 01 00 	movq   $0x0,0x1f8(%rsp)
  4022ed:	00 00 00 00 00 
  4022f2:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
  4022f9:	00 00 00 00 00 
  4022fe:	48 89 84 24 88 02 00 	mov    %rax,0x288(%rsp)
  402305:	00 
  402306:	48 c7 84 24 98 02 00 	movq   $0x0,0x298(%rsp)
  40230d:	00 00 00 00 00 
  402312:	48 c7 84 24 90 02 00 	movq   $0x0,0x290(%rsp)
  402319:	00 00 00 00 00 
  40231e:	c7 84 24 40 02 00 00 	movl   $0x42ff0000,0x240(%rsp)
  402325:	00 00 ff 42 
  402329:	c7 84 24 4c 02 00 00 	movl   $0x0,0x24c(%rsp)
  402330:	00 00 00 00 
  402334:	c7 84 24 48 02 00 00 	movl   $0x0,0x248(%rsp)
  40233b:	00 00 00 00 
  40233f:	c7 84 24 44 02 00 00 	movl   $0x0,0x244(%rsp)
  402346:	00 00 00 00 
  40234a:	48 c7 84 24 70 02 00 	movq   $0x0,0x270(%rsp)
  402351:	00 00 00 00 00 
  402356:	48 c7 84 24 68 02 00 	movq   $0x0,0x268(%rsp)
  40235d:	00 00 00 00 00 
  402362:	48 c7 84 24 60 02 00 	movq   $0x0,0x260(%rsp)
  402369:	00 00 00 00 00 
  40236e:	48 c7 84 24 50 02 00 	movq   $0x0,0x250(%rsp)
  402375:	00 00 00 00 00 
  40237a:	48 c7 84 24 58 02 00 	movq   $0x0,0x258(%rsp)
  402381:	00 00 00 00 00 
  402386:	48 c7 84 24 78 02 00 	movq   $0x0,0x278(%rsp)
  40238d:	00 00 00 00 00 
  402392:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  402399:	00 00 
  40239b:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  4023a2:	00 00 
  4023a4:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  4023ab:	00 00 
  4023ad:	e8 fe ee ff ff       	callq  4012b0 <_ZNSsC1EPKcRKSaIcE@plt>
  4023b2:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  4023b7:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  4023be:	00 
  4023bf:	ba 01 00 00 00       	mov    $0x1,%edx
  4023c4:	e8 d7 ee ff ff       	callq  4012a0 <_ZN2cv6imreadERKSsi@plt>
  4023c9:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  4023d0:	00 
  4023d1:	48 85 c0             	test   %rax,%rax
  4023d4:	74 04                	je     4023da <_Z11decodeImagePfS_S_+0x3ba>
  4023d6:	f0 83 00 01          	lock addl $0x1,(%rax)
  4023da:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  4023df:	48 85 c0             	test   %rax,%rax
  4023e2:	74 0a                	je     4023ee <_Z11decodeImagePfS_S_+0x3ce>
  4023e4:	f0 83 28 01          	lock subl $0x1,(%rax)
  4023e8:	0f 84 d2 03 00 00    	je     4027c0 <_Z11decodeImagePfS_S_+0x7a0>
  4023ee:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  4023f5:	00 
  4023f6:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  4023fd:	00 00 00 00 00 
  402402:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  402409:	00 00 00 00 00 
  40240e:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402415:	00 00 00 00 00 
  40241a:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402421:	00 00 
  402423:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402429:	83 7c 24 64 02       	cmpl   $0x2,0x64(%rsp)
  40242e:	8b 84 24 20 01 00 00 	mov    0x120(%rsp),%eax
  402435:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  40243c:	00 00 
  40243e:	89 44 24 60          	mov    %eax,0x60(%rsp)
  402442:	7f 10                	jg     402454 <_Z11decodeImagePfS_S_+0x434>
  402444:	8b 84 24 24 01 00 00 	mov    0x124(%rsp),%eax
  40244b:	83 f8 02             	cmp    $0x2,%eax
  40244e:	0f 8e 1c 03 00 00    	jle    402770 <_Z11decodeImagePfS_S_+0x750>
  402454:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  40245b:	00 
  40245c:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402461:	e8 9a ec ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402466:	c5 f9 6f 84 24 30 01 	vmovdqa 0x130(%rsp),%xmm0
  40246d:	00 00 
  40246f:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  402476:	00 
  402477:	c5 f9 7f 44 24 70    	vmovdqa %xmm0,0x70(%rsp)
  40247d:	48 85 c0             	test   %rax,%rax
  402480:	c5 f9 6f 84 24 40 01 	vmovdqa 0x140(%rsp),%xmm0
  402487:	00 00 
  402489:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  402490:	00 00 
  402492:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  402499:	00 00 
  40249b:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  4024a2:	00 00 
  4024a4:	74 0a                	je     4024b0 <_Z11decodeImagePfS_S_+0x490>
  4024a6:	f0 83 28 01          	lock subl $0x1,(%rax)
  4024aa:	0f 84 20 03 00 00    	je     4027d0 <_Z11decodeImagePfS_S_+0x7b0>
  4024b0:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  4024b7:	00 
  4024b8:	48 8b bc 24 68 01 00 	mov    0x168(%rsp),%rdi
  4024bf:	00 
  4024c0:	48 c7 84 24 50 01 00 	movq   $0x0,0x150(%rsp)
  4024c7:	00 00 00 00 00 
  4024cc:	48 c7 84 24 48 01 00 	movq   $0x0,0x148(%rsp)
  4024d3:	00 00 00 00 00 
  4024d8:	48 c7 84 24 40 01 00 	movq   $0x0,0x140(%rsp)
  4024df:	00 00 00 00 00 
  4024e4:	48 c7 84 24 30 01 00 	movq   $0x0,0x130(%rsp)
  4024eb:	00 00 00 00 00 
  4024f0:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4024f6:	48 8d 84 24 70 01 00 	lea    0x170(%rsp),%rax
  4024fd:	00 
  4024fe:	48 c7 84 24 38 01 00 	movq   $0x0,0x138(%rsp)
  402505:	00 00 00 00 00 
  40250a:	48 39 c7             	cmp    %rax,%rdi
  40250d:	74 05                	je     402514 <_Z11decodeImagePfS_S_+0x4f4>
  40250f:	e8 2c ed ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  402514:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  402519:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  40251d:	48 81 ff 20 51 60 00 	cmp    $0x605120,%rdi
  402524:	0f 85 e5 02 00 00    	jne    40280f <_Z11decodeImagePfS_S_+0x7ef>
  40252a:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  40252f:	48 89 de             	mov    %rbx,%rsi
  402532:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  402537:	44 8b 7c 24 6c       	mov    0x6c(%rsp),%r15d
  40253c:	e8 4f ed ff ff       	callq  401290 <_ZN2cv5splitERKNS_3MatEPS0_@plt>
  402541:	48 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%rdx
  402548:	00 
  402549:	48 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%rsi
  402550:	00 
  402551:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402556:	c6 04 24 00          	movb   $0x0,(%rsp)
  40255a:	e8 91 0e 00 00       	callq  4033f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  40255f:	45 0f af f7          	imul   %r15d,%r14d
  402563:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402568:	4c 89 ef             	mov    %r13,%rdi
  40256b:	4d 63 f6             	movslq %r14d,%r14
  40256e:	49 c1 e6 02          	shl    $0x2,%r14
  402572:	4c 89 f2             	mov    %r14,%rdx
  402575:	e8 76 eb ff ff       	callq  4010f0 <memcpy@plt>
  40257a:	48 8b 94 24 08 02 00 	mov    0x208(%rsp),%rdx
  402581:	00 
  402582:	48 8b b4 24 00 02 00 	mov    0x200(%rsp),%rsi
  402589:	00 
  40258a:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  40258f:	c6 04 24 00          	movb   $0x0,(%rsp)
  402593:	e8 58 0e 00 00       	callq  4033f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402598:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40259d:	4c 89 f2             	mov    %r14,%rdx
  4025a0:	4c 89 e7             	mov    %r12,%rdi
  4025a3:	e8 48 eb ff ff       	callq  4010f0 <memcpy@plt>
  4025a8:	48 8b 94 24 68 02 00 	mov    0x268(%rsp),%rdx
  4025af:	00 
  4025b0:	48 8b b4 24 60 02 00 	mov    0x260(%rsp),%rsi
  4025b7:	00 
  4025b8:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  4025bd:	c6 04 24 00          	movb   $0x0,(%rsp)
  4025c1:	e8 2a 0e 00 00       	callq  4033f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  4025c6:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
  4025cb:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4025d0:	4c 89 f2             	mov    %r14,%rdx
  4025d3:	4c 89 e6             	mov    %r12,%rsi
  4025d6:	e8 15 eb ff ff       	callq  4010f0 <memcpy@plt>
  4025db:	4d 85 e4             	test   %r12,%r12
  4025de:	74 08                	je     4025e8 <_Z11decodeImagePfS_S_+0x5c8>
  4025e0:	4c 89 e7             	mov    %r12,%rdi
  4025e3:	e8 88 eb ff ff       	callq  401170 <_ZdlPv@plt>
  4025e8:	4c 8d a4 24 a0 02 00 	lea    0x2a0(%rsp),%r12
  4025ef:	00 
  4025f0:	49 83 ec 60          	sub    $0x60,%r12
  4025f4:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  4025f9:	48 85 c0             	test   %rax,%rax
  4025fc:	74 0a                	je     402608 <_Z11decodeImagePfS_S_+0x5e8>
  4025fe:	f0 83 28 01          	lock subl $0x1,(%rax)
  402602:	0f 84 58 01 00 00    	je     402760 <_Z11decodeImagePfS_S_+0x740>
  402608:	49 8b 44 24 40       	mov    0x40(%r12),%rax
  40260d:	49 c7 44 24 30 00 00 	movq   $0x0,0x30(%r12)
  402614:	00 00 
  402616:	49 c7 44 24 28 00 00 	movq   $0x0,0x28(%r12)
  40261d:	00 00 
  40261f:	49 c7 44 24 20 00 00 	movq   $0x0,0x20(%r12)
  402626:	00 00 
  402628:	49 c7 44 24 10 00 00 	movq   $0x0,0x10(%r12)
  40262f:	00 00 
  402631:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402637:	49 8b 7c 24 48       	mov    0x48(%r12),%rdi
  40263c:	49 8d 44 24 50       	lea    0x50(%r12),%rax
  402641:	49 c7 44 24 18 00 00 	movq   $0x0,0x18(%r12)
  402648:	00 00 
  40264a:	48 39 c7             	cmp    %rax,%rdi
  40264d:	74 05                	je     402654 <_Z11decodeImagePfS_S_+0x634>
  40264f:	e8 ec eb ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  402654:	49 39 dc             	cmp    %rbx,%r12
  402657:	75 97                	jne    4025f0 <_Z11decodeImagePfS_S_+0x5d0>
  402659:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  402660:	00 
  402661:	48 85 c0             	test   %rax,%rax
  402664:	74 0a                	je     402670 <_Z11decodeImagePfS_S_+0x650>
  402666:	f0 83 28 01          	lock subl $0x1,(%rax)
  40266a:	0f 84 78 01 00 00    	je     4027e8 <_Z11decodeImagePfS_S_+0x7c8>
  402670:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  402677:	00 
  402678:	48 8b bc 24 08 01 00 	mov    0x108(%rsp),%rdi
  40267f:	00 
  402680:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402687:	00 00 00 00 00 
  40268c:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  402693:	00 00 00 00 00 
  402698:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  40269f:	00 00 00 00 00 
  4026a4:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  4026ab:	00 00 00 00 00 
  4026b0:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4026b6:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  4026bd:	00 
  4026be:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  4026c5:	00 00 00 00 00 
  4026ca:	48 39 c7             	cmp    %rax,%rdi
  4026cd:	74 05                	je     4026d4 <_Z11decodeImagePfS_S_+0x6b4>
  4026cf:	e8 6c eb ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  4026d4:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  4026d9:	48 85 c0             	test   %rax,%rax
  4026dc:	74 0a                	je     4026e8 <_Z11decodeImagePfS_S_+0x6c8>
  4026de:	f0 83 28 01          	lock subl $0x1,(%rax)
  4026e2:	0f 84 18 01 00 00    	je     402800 <_Z11decodeImagePfS_S_+0x7e0>
  4026e8:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  4026ef:	00 
  4026f0:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  4026f7:	00 
  4026f8:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  4026ff:	00 00 00 00 00 
  402704:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  40270b:	00 00 00 00 00 
  402710:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402717:	00 00 00 00 00 
  40271c:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402723:	00 00 
  402725:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40272b:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  402732:	00 
  402733:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  40273a:	00 00 
  40273c:	48 39 c7             	cmp    %rax,%rdi
  40273f:	74 05                	je     402746 <_Z11decodeImagePfS_S_+0x726>
  402741:	e8 fa ea ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  402746:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  40274a:	31 c0                	xor    %eax,%eax
  40274c:	5b                   	pop    %rbx
  40274d:	41 5c                	pop    %r12
  40274f:	41 5d                	pop    %r13
  402751:	41 5e                	pop    %r14
  402753:	41 5f                	pop    %r15
  402755:	5d                   	pop    %rbp
  402756:	c3                   	retq   
  402757:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40275e:	00 00 
  402760:	4c 89 e7             	mov    %r12,%rdi
  402763:	e8 f8 e9 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  402768:	e9 9b fe ff ff       	jmpq   402608 <_Z11decodeImagePfS_S_+0x5e8>
  40276d:	0f 1f 00             	nopl   (%rax)
  402770:	89 44 24 64          	mov    %eax,0x64(%rsp)
  402774:	8b 84 24 28 01 00 00 	mov    0x128(%rsp),%eax
  40277b:	89 44 24 68          	mov    %eax,0x68(%rsp)
  40277f:	8b 84 24 2c 01 00 00 	mov    0x12c(%rsp),%eax
  402786:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
  40278a:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402791:	00 
  402792:	48 8b 10             	mov    (%rax),%rdx
  402795:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  40279c:	00 
  40279d:	48 89 10             	mov    %rdx,(%rax)
  4027a0:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  4027a7:	00 
  4027a8:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4027ac:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  4027b3:	00 
  4027b4:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4027b8:	e9 a9 fc ff ff       	jmpq   402466 <_Z11decodeImagePfS_S_+0x446>
  4027bd:	0f 1f 00             	nopl   (%rax)
  4027c0:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  4027c5:	e8 96 e9 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  4027ca:	e9 1f fc ff ff       	jmpq   4023ee <_Z11decodeImagePfS_S_+0x3ce>
  4027cf:	90                   	nop
  4027d0:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  4027d7:	00 
  4027d8:	e8 83 e9 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  4027dd:	e9 ce fc ff ff       	jmpq   4024b0 <_Z11decodeImagePfS_S_+0x490>
  4027e2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4027e8:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  4027ef:	00 
  4027f0:	e8 6b e9 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  4027f5:	e9 76 fe ff ff       	jmpq   402670 <_Z11decodeImagePfS_S_+0x650>
  4027fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402800:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402805:	e8 56 e9 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  40280a:	e9 d9 fe ff ff       	jmpq   4026e8 <_Z11decodeImagePfS_S_+0x6c8>
  40280f:	b9 50 12 40 00       	mov    $0x401250,%ecx
  402814:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  402818:	48 85 c9             	test   %rcx,%rcx
  40281b:	74 46                	je     402863 <_Z11decodeImagePfS_S_+0x843>
  40281d:	83 c8 ff             	or     $0xffffffff,%eax
  402820:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  402824:	85 c0                	test   %eax,%eax
  402826:	0f 8f fe fc ff ff    	jg     40252a <_Z11decodeImagePfS_S_+0x50a>
  40282c:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  402833:	00 
  402834:	e8 37 ea ff ff       	callq  401270 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  402839:	e9 ec fc ff ff       	jmpq   40252a <_Z11decodeImagePfS_S_+0x50a>
  40283e:	48 89 c3             	mov    %rax,%rbx
  402841:	c5 f8 77             	vzeroupper 
  402844:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  40284b:	00 
  40284c:	e8 8f 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  402851:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402856:	e8 85 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  40285b:	48 89 df             	mov    %rbx,%rdi
  40285e:	e8 4d e9 ff ff       	callq  4011b0 <_Unwind_Resume@plt>
  402863:	8b 50 f8             	mov    -0x8(%rax),%edx
  402866:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  402869:	89 48 f8             	mov    %ecx,-0x8(%rax)
  40286c:	89 d0                	mov    %edx,%eax
  40286e:	eb b4                	jmp    402824 <_Z11decodeImagePfS_S_+0x804>
  402870:	49 89 c4             	mov    %rax,%r12
  402873:	c5 f8 77             	vzeroupper 
  402876:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  40287b:	48 8d 74 24 1f       	lea    0x1f(%rsp),%rsi
  402880:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402884:	e8 77 e9 ff ff       	callq  401200 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  402889:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  40288e:	48 85 ff             	test   %rdi,%rdi
  402891:	74 05                	je     402898 <_Z11decodeImagePfS_S_+0x878>
  402893:	e8 d8 e8 ff ff       	callq  401170 <_ZdlPv@plt>
  402898:	48 8d bb c0 00 00 00 	lea    0xc0(%rbx),%rdi
  40289f:	e8 3c 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4028a4:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  4028a8:	e8 33 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4028ad:	48 89 df             	mov    %rbx,%rdi
  4028b0:	4c 89 e3             	mov    %r12,%rbx
  4028b3:	e8 28 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4028b8:	eb 8a                	jmp    402844 <_Z11decodeImagePfS_S_+0x824>
  4028ba:	49 89 c4             	mov    %rax,%r12
  4028bd:	c5 f8 77             	vzeroupper 
  4028c0:	eb c7                	jmp    402889 <_Z11decodeImagePfS_S_+0x869>
  4028c2:	48 89 c3             	mov    %rax,%rbx
  4028c5:	c5 f8 77             	vzeroupper 
  4028c8:	eb 87                	jmp    402851 <_Z11decodeImagePfS_S_+0x831>
  4028ca:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  4028d1:	00 
  4028d2:	49 89 c4             	mov    %rax,%r12
  4028d5:	c5 f8 77             	vzeroupper 
  4028d8:	e8 03 00 00 00       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4028dd:	eb 97                	jmp    402876 <_Z11decodeImagePfS_S_+0x856>
  4028df:	90                   	nop

00000000004028e0 <_ZN2cv3MatD1Ev>:
  4028e0:	53                   	push   %rbx
  4028e1:	48 8b 47 18          	mov    0x18(%rdi),%rax
  4028e5:	48 89 fb             	mov    %rdi,%rbx
  4028e8:	48 85 c0             	test   %rax,%rax
  4028eb:	74 06                	je     4028f3 <_ZN2cv3MatD1Ev+0x13>
  4028ed:	f0 83 28 01          	lock subl $0x1,(%rax)
  4028f1:	74 55                	je     402948 <_ZN2cv3MatD1Ev+0x68>
  4028f3:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4028f7:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  4028fb:	48 83 c3 50          	add    $0x50,%rbx
  4028ff:	48 c7 43 e0 00 00 00 	movq   $0x0,-0x20(%rbx)
  402906:	00 
  402907:	48 c7 43 d8 00 00 00 	movq   $0x0,-0x28(%rbx)
  40290e:	00 
  40290f:	48 c7 43 d0 00 00 00 	movq   $0x0,-0x30(%rbx)
  402916:	00 
  402917:	48 c7 43 c0 00 00 00 	movq   $0x0,-0x40(%rbx)
  40291e:	00 
  40291f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402925:	48 c7 43 c8 00 00 00 	movq   $0x0,-0x38(%rbx)
  40292c:	00 
  40292d:	48 39 df             	cmp    %rbx,%rdi
  402930:	74 0e                	je     402940 <_ZN2cv3MatD1Ev+0x60>
  402932:	5b                   	pop    %rbx
  402933:	e9 08 e9 ff ff       	jmpq   401240 <_ZN2cv8fastFreeEPv@plt>
  402938:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40293f:	00 
  402940:	5b                   	pop    %rbx
  402941:	c3                   	retq   
  402942:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402948:	e8 13 e8 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  40294d:	0f 1f 00             	nopl   (%rax)
  402950:	eb a1                	jmp    4028f3 <_ZN2cv3MatD1Ev+0x13>
  402952:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  402959:	00 00 00 
  40295c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000402960 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>:
  402960:	41 54                	push   %r12
  402962:	49 89 fc             	mov    %rdi,%r12
  402965:	55                   	push   %rbp
  402966:	53                   	push   %rbx
  402967:	48 8b 6f 08          	mov    0x8(%rdi),%rbp
  40296b:	48 8b 1f             	mov    (%rdi),%rbx
  40296e:	48 39 dd             	cmp    %rbx,%rbp
  402971:	75 5a                	jne    4029cd <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x6d>
  402973:	e9 95 00 00 00       	jmpq   402a0d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xad>
  402978:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40297f:	00 
  402980:	48 8b 43 40          	mov    0x40(%rbx),%rax
  402984:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  40298b:	00 
  40298c:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  402993:	00 
  402994:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40299b:	00 
  40299c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  4029a3:	00 
  4029a4:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4029aa:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  4029ae:	48 8d 43 50          	lea    0x50(%rbx),%rax
  4029b2:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  4029b9:	00 
  4029ba:	48 39 c7             	cmp    %rax,%rdi
  4029bd:	74 05                	je     4029c4 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x64>
  4029bf:	e8 7c e8 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  4029c4:	48 83 c3 60          	add    $0x60,%rbx
  4029c8:	48 39 dd             	cmp    %rbx,%rbp
  4029cb:	74 23                	je     4029f0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x90>
  4029cd:	48 8b 43 18          	mov    0x18(%rbx),%rax
  4029d1:	48 85 c0             	test   %rax,%rax
  4029d4:	74 aa                	je     402980 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4029d6:	f0 83 28 01          	lock subl $0x1,(%rax)
  4029da:	75 a4                	jne    402980 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4029dc:	48 89 df             	mov    %rbx,%rdi
  4029df:	e8 7c e7 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  4029e4:	eb 9a                	jmp    402980 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4029e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4029ed:	00 00 00 
  4029f0:	49 8b 3c 24          	mov    (%r12),%rdi
  4029f4:	48 85 ff             	test   %rdi,%rdi
  4029f7:	74 0f                	je     402a08 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xa8>
  4029f9:	5b                   	pop    %rbx
  4029fa:	5d                   	pop    %rbp
  4029fb:	41 5c                	pop    %r12
  4029fd:	e9 6e e7 ff ff       	jmpq   401170 <_ZdlPv@plt>
  402a02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402a08:	5b                   	pop    %rbx
  402a09:	5d                   	pop    %rbp
  402a0a:	41 5c                	pop    %r12
  402a0c:	c3                   	retq   
  402a0d:	48 89 ef             	mov    %rbp,%rdi
  402a10:	eb e2                	jmp    4029f4 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x94>
  402a12:	49 8b 3c 24          	mov    (%r12),%rdi
  402a16:	48 89 c3             	mov    %rax,%rbx
  402a19:	48 85 ff             	test   %rdi,%rdi
  402a1c:	74 10                	je     402a2e <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xce>
  402a1e:	c5 f8 77             	vzeroupper 
  402a21:	e8 4a e7 ff ff       	callq  401170 <_ZdlPv@plt>
  402a26:	48 89 df             	mov    %rbx,%rdi
  402a29:	e8 82 e7 ff ff       	callq  4011b0 <_Unwind_Resume@plt>
  402a2e:	c5 f8 77             	vzeroupper 
  402a31:	eb f3                	jmp    402a26 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xc6>
  402a33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  402a3a:	00 00 00 
  402a3d:	0f 1f 00             	nopl   (%rax)

0000000000402a40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>:
  402a40:	55                   	push   %rbp
  402a41:	48 89 e5             	mov    %rsp,%rbp
  402a44:	41 57                	push   %r15
  402a46:	49 89 ff             	mov    %rdi,%r15
  402a49:	41 56                	push   %r14
  402a4b:	49 89 f6             	mov    %rsi,%r14
  402a4e:	41 55                	push   %r13
  402a50:	41 54                	push   %r12
  402a52:	53                   	push   %rbx
  402a53:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  402a57:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  402a5e:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  402a62:	49 3b 7f 10          	cmp    0x10(%r15),%rdi
  402a66:	0f 84 3b 04 00 00    	je     402ea7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x467>
  402a6c:	31 c0                	xor    %eax,%eax
  402a6e:	48 85 ff             	test   %rdi,%rdi
  402a71:	0f 84 8c 00 00 00    	je     402b03 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  402a77:	48 8b 4f c0          	mov    -0x40(%rdi),%rcx
  402a7b:	48 8b 47 b0          	mov    -0x50(%rdi),%rax
  402a7f:	c5 fa 6f 47 a0       	vmovdqu -0x60(%rdi),%xmm0
  402a84:	48 89 4f 20          	mov    %rcx,0x20(%rdi)
  402a88:	48 8b 4f c8          	mov    -0x38(%rdi),%rcx
  402a8c:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402a90:	48 8b 47 b8          	mov    -0x48(%rdi),%rax
  402a94:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402a98:	48 89 4f 28          	mov    %rcx,0x28(%rdi)
  402a9c:	48 8b 4f d0          	mov    -0x30(%rdi),%rcx
  402aa0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402aa4:	48 85 c0             	test   %rax,%rax
  402aa7:	48 89 4f 30          	mov    %rcx,0x30(%rdi)
  402aab:	48 8b 4f d8          	mov    -0x28(%rdi),%rcx
  402aaf:	48 89 47 18          	mov    %rax,0x18(%rdi)
  402ab3:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402ab8:	48 89 4f 38          	mov    %rcx,0x38(%rdi)
  402abc:	48 8d 4f 08          	lea    0x8(%rdi),%rcx
  402ac0:	48 89 4f 40          	mov    %rcx,0x40(%rdi)
  402ac4:	48 8d 4f 50          	lea    0x50(%rdi),%rcx
  402ac8:	48 89 4f 48          	mov    %rcx,0x48(%rdi)
  402acc:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  402acf:	74 07                	je     402ad8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x98>
  402ad1:	f0 83 00 01          	lock addl $0x1,(%rax)
  402ad5:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  402ad8:	83 f9 02             	cmp    $0x2,%ecx
  402adb:	0f 8f 57 03 00 00    	jg     402e38 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3f8>
  402ae1:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  402ae5:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402ae9:	48 8b 09             	mov    (%rcx),%rcx
  402aec:	48 89 08             	mov    %rcx,(%rax)
  402aef:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  402af3:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402af7:	48 8b 49 08          	mov    0x8(%rcx),%rcx
  402afb:	48 89 48 08          	mov    %rcx,0x8(%rax)
  402aff:	49 8b 47 08          	mov    0x8(%r15),%rax
  402b03:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  402b07:	48 83 c0 60          	add    $0x60,%rax
  402b0b:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  402b10:	49 89 47 08          	mov    %rax,0x8(%r15)
  402b14:	48 8b 42 10          	mov    0x10(%rdx),%rax
  402b18:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  402b1c:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  402b21:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  402b25:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  402b2a:	48 8b 42 18          	mov    0x18(%rdx),%rax
  402b2e:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
  402b34:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  402b39:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  402b3d:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402b41:	48 85 c0             	test   %rax,%rax
  402b44:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  402b49:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  402b4d:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  402b52:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  402b59:	00 00 
  402b5b:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  402b60:	49 8d 4d 08          	lea    0x8(%r13),%rcx
  402b64:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
  402b69:	49 8d 4d 50          	lea    0x50(%r13),%rcx
  402b6d:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
  402b72:	8b 4a 04             	mov    0x4(%rdx),%ecx
  402b75:	74 07                	je     402b7e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x13e>
  402b77:	f0 83 00 01          	lock addl $0x1,(%rax)
  402b7b:	8b 4a 04             	mov    0x4(%rdx),%ecx
  402b7e:	83 f9 02             	cmp    $0x2,%ecx
  402b81:	0f 8f d9 02 00 00    	jg     402e60 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x420>
  402b87:	48 8b 42 48          	mov    0x48(%rdx),%rax
  402b8b:	48 8b 08             	mov    (%rax),%rcx
  402b8e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  402b93:	48 89 08             	mov    %rcx,(%rax)
  402b96:	48 8b 42 48          	mov    0x48(%rdx),%rax
  402b9a:	48 8b 50 08          	mov    0x8(%rax),%rdx
  402b9e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  402ba3:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402ba7:	4d 8b 67 08          	mov    0x8(%r15),%r12
  402bab:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  402bb2:	aa aa aa 
  402bb5:	49 8d 5c 24 a0       	lea    -0x60(%r12),%rbx
  402bba:	49 81 ec c0 00 00 00 	sub    $0xc0,%r12
  402bc1:	4d 29 f4             	sub    %r14,%r12
  402bc4:	49 c1 fc 05          	sar    $0x5,%r12
  402bc8:	4c 0f af e0          	imul   %rax,%r12
  402bcc:	4d 85 e4             	test   %r12,%r12
  402bcf:	7f 76                	jg     402c47 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x207>
  402bd1:	e9 0a 01 00 00       	jmpq   402ce0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  402bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  402bdd:	00 00 00 
  402be0:	89 43 04             	mov    %eax,0x4(%rbx)
  402be3:	8b 43 a8             	mov    -0x58(%rbx),%eax
  402be6:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  402bea:	89 43 08             	mov    %eax,0x8(%rbx)
  402bed:	8b 43 ac             	mov    -0x54(%rbx),%eax
  402bf0:	89 43 0c             	mov    %eax,0xc(%rbx)
  402bf3:	48 8b 43 48          	mov    0x48(%rbx),%rax
  402bf7:	48 8b 12             	mov    (%rdx),%rdx
  402bfa:	48 89 10             	mov    %rdx,(%rax)
  402bfd:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  402c01:	48 8b 43 48          	mov    0x48(%rbx),%rax
  402c05:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402c09:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402c0d:	48 8b 43 b0          	mov    -0x50(%rbx),%rax
  402c11:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402c15:	48 8b 43 c0          	mov    -0x40(%rbx),%rax
  402c19:	48 89 43 20          	mov    %rax,0x20(%rbx)
  402c1d:	48 8b 43 c8          	mov    -0x38(%rbx),%rax
  402c21:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402c25:	48 8b 43 d0          	mov    -0x30(%rbx),%rax
  402c29:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402c2d:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  402c31:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402c35:	48 8b 43 d8          	mov    -0x28(%rbx),%rax
  402c39:	48 89 43 38          	mov    %rax,0x38(%rbx)
  402c3d:	49 83 ec 01          	sub    $0x1,%r12
  402c41:	0f 84 99 00 00 00    	je     402ce0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  402c47:	4c 8d bb 40 ff ff ff 	lea    -0xc0(%rbx),%r15
  402c4e:	48 83 eb 60          	sub    $0x60,%rbx
  402c52:	4c 39 fb             	cmp    %r15,%rbx
  402c55:	74 e6                	je     402c3d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1fd>
  402c57:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  402c5b:	48 85 c0             	test   %rax,%rax
  402c5e:	74 04                	je     402c64 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x224>
  402c60:	f0 83 00 01          	lock addl $0x1,(%rax)
  402c64:	48 8b 43 18          	mov    0x18(%rbx),%rax
  402c68:	48 85 c0             	test   %rax,%rax
  402c6b:	74 06                	je     402c73 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  402c6d:	f0 83 28 01          	lock subl $0x1,(%rax)
  402c71:	74 5d                	je     402cd0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x290>
  402c73:	48 8b 43 40          	mov    0x40(%rbx),%rax
  402c77:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  402c7e:	00 
  402c7f:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  402c86:	00 
  402c87:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  402c8e:	00 
  402c8f:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  402c96:	00 
  402c97:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402c9d:	83 7b 04 02          	cmpl   $0x2,0x4(%rbx)
  402ca1:	8b 43 a0             	mov    -0x60(%rbx),%eax
  402ca4:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  402cab:	00 
  402cac:	89 03                	mov    %eax,(%rbx)
  402cae:	7f 0c                	jg     402cbc <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x27c>
  402cb0:	8b 43 a4             	mov    -0x5c(%rbx),%eax
  402cb3:	83 f8 02             	cmp    $0x2,%eax
  402cb6:	0f 8e 24 ff ff ff    	jle    402be0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1a0>
  402cbc:	4c 89 fe             	mov    %r15,%rsi
  402cbf:	48 89 df             	mov    %rbx,%rdi
  402cc2:	e8 39 e4 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402cc7:	e9 41 ff ff ff       	jmpq   402c0d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1cd>
  402ccc:	0f 1f 40 00          	nopl   0x0(%rax)
  402cd0:	48 89 df             	mov    %rbx,%rdi
  402cd3:	e8 88 e4 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  402cd8:	eb 99                	jmp    402c73 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  402cda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402ce0:	4d 39 ee             	cmp    %r13,%r14
  402ce3:	0f 84 b4 01 00 00    	je     402e9d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x45d>
  402ce9:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  402cee:	48 85 c0             	test   %rax,%rax
  402cf1:	74 04                	je     402cf7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2b7>
  402cf3:	f0 83 00 01          	lock addl $0x1,(%rax)
  402cf7:	49 8b 46 18          	mov    0x18(%r14),%rax
  402cfb:	48 85 c0             	test   %rax,%rax
  402cfe:	74 0a                	je     402d0a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  402d00:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d04:	0f 84 86 01 00 00    	je     402e90 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x450>
  402d0a:	49 8b 46 40          	mov    0x40(%r14),%rax
  402d0e:	49 c7 46 30 00 00 00 	movq   $0x0,0x30(%r14)
  402d15:	00 
  402d16:	49 c7 46 28 00 00 00 	movq   $0x0,0x28(%r14)
  402d1d:	00 
  402d1e:	49 c7 46 20 00 00 00 	movq   $0x0,0x20(%r14)
  402d25:	00 
  402d26:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  402d2d:	00 
  402d2e:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402d34:	41 83 7e 04 02       	cmpl   $0x2,0x4(%r14)
  402d39:	8b 44 24 30          	mov    0x30(%rsp),%eax
  402d3d:	49 c7 46 18 00 00 00 	movq   $0x0,0x18(%r14)
  402d44:	00 
  402d45:	41 89 06             	mov    %eax,(%r14)
  402d48:	7f 0d                	jg     402d57 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x317>
  402d4a:	8b 44 24 34          	mov    0x34(%rsp),%eax
  402d4e:	83 f8 02             	cmp    $0x2,%eax
  402d51:	0f 8e a1 00 00 00    	jle    402df8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3b8>
  402d57:	4c 89 ee             	mov    %r13,%rsi
  402d5a:	4c 89 f7             	mov    %r14,%rdi
  402d5d:	e8 9e e3 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402d62:	c5 f9 6f 44 24 40    	vmovdqa 0x40(%rsp),%xmm0
  402d68:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  402d6d:	c4 c1 7a 7f 46 10    	vmovdqu %xmm0,0x10(%r14)
  402d73:	c5 f9 6f 44 24 50    	vmovdqa 0x50(%rsp),%xmm0
  402d79:	c4 c1 7a 7f 46 20    	vmovdqu %xmm0,0x20(%r14)
  402d7f:	c5 f9 6f 44 24 60    	vmovdqa 0x60(%rsp),%xmm0
  402d85:	c4 c1 7a 7f 46 30    	vmovdqu %xmm0,0x30(%r14)
  402d8b:	48 85 c0             	test   %rax,%rax
  402d8e:	74 0a                	je     402d9a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  402d90:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d94:	0f 84 e6 00 00 00    	je     402e80 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x440>
  402d9a:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  402d9f:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  402da4:	49 83 c5 50          	add    $0x50,%r13
  402da8:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  402daf:	00 00 
  402db1:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  402db8:	00 00 
  402dba:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402dc1:	00 00 
  402dc3:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  402dca:	00 00 
  402dcc:	4c 39 ef             	cmp    %r13,%rdi
  402dcf:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402dd5:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  402ddc:	00 00 
  402dde:	74 05                	je     402de5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  402de0:	e8 5b e4 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  402de5:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  402de9:	5b                   	pop    %rbx
  402dea:	41 5c                	pop    %r12
  402dec:	41 5d                	pop    %r13
  402dee:	41 5e                	pop    %r14
  402df0:	41 5f                	pop    %r15
  402df2:	5d                   	pop    %rbp
  402df3:	c3                   	retq   
  402df4:	0f 1f 40 00          	nopl   0x0(%rax)
  402df8:	41 89 46 04          	mov    %eax,0x4(%r14)
  402dfc:	8b 44 24 38          	mov    0x38(%rsp),%eax
  402e00:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  402e05:	41 89 46 08          	mov    %eax,0x8(%r14)
  402e09:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  402e0d:	48 8b 12             	mov    (%rdx),%rdx
  402e10:	41 89 46 0c          	mov    %eax,0xc(%r14)
  402e14:	49 8b 46 48          	mov    0x48(%r14),%rax
  402e18:	48 89 10             	mov    %rdx,(%rax)
  402e1b:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  402e20:	49 8b 46 48          	mov    0x48(%r14),%rax
  402e24:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402e28:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402e2c:	e9 31 ff ff ff       	jmpq   402d62 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x322>
  402e31:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  402e38:	48 8d 77 a0          	lea    -0x60(%rdi),%rsi
  402e3c:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  402e43:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  402e48:	e8 b3 e2 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402e4d:	49 8b 47 08          	mov    0x8(%r15),%rax
  402e51:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  402e56:	e9 a8 fc ff ff       	jmpq   402b03 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  402e5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402e60:	48 89 d6             	mov    %rdx,%rsi
  402e63:	4c 89 ef             	mov    %r13,%rdi
  402e66:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
  402e6d:	00 
  402e6e:	e8 8d e2 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402e73:	e9 2f fd ff ff       	jmpq   402ba7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x167>
  402e78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  402e7f:	00 
  402e80:	4c 89 ef             	mov    %r13,%rdi
  402e83:	e8 d8 e2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  402e88:	e9 0d ff ff ff       	jmpq   402d9a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  402e8d:	0f 1f 00             	nopl   (%rax)
  402e90:	4c 89 f7             	mov    %r14,%rdi
  402e93:	e8 c8 e2 ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  402e98:	e9 6d fe ff ff       	jmpq   402d0a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  402e9d:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  402ea2:	e9 e4 fe ff ff       	jmpq   402d8b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x34b>
  402ea7:	49 8b 07             	mov    (%r15),%rax
  402eaa:	48 b9 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rcx
  402eb1:	aa aa aa 
  402eb4:	48 29 c7             	sub    %rax,%rdi
  402eb7:	48 c1 ff 05          	sar    $0x5,%rdi
  402ebb:	48 0f af f9          	imul   %rcx,%rdi
  402ebf:	48 85 ff             	test   %rdi,%rdi
  402ec2:	0f 85 be 01 00 00    	jne    403086 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x646>
  402ec8:	48 89 f3             	mov    %rsi,%rbx
  402ecb:	be 01 00 00 00       	mov    $0x1,%esi
  402ed0:	48 29 c3             	sub    %rax,%rbx
  402ed3:	48 c1 fb 05          	sar    $0x5,%rbx
  402ed7:	48 0f af d9          	imul   %rcx,%rbx
  402edb:	48 8d 3c 76          	lea    (%rsi,%rsi,2),%rdi
  402edf:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  402ee4:	48 c1 e7 05          	shl    $0x5,%rdi
  402ee8:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  402eed:	e8 2e e2 ff ff       	callq  401120 <_Znwm@plt>
  402ef2:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  402ef7:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  402efc:	48 8d 04 5b          	lea    (%rbx,%rbx,2),%rax
  402f00:	48 c1 e0 05          	shl    $0x5,%rax
  402f04:	48 03 44 24 18       	add    0x18(%rsp),%rax
  402f09:	48 89 c6             	mov    %rax,%rsi
  402f0c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  402f11:	0f 84 87 00 00 00    	je     402f9e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  402f17:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  402f1b:	c5 fa 7f 00          	vmovdqu %xmm0,(%rax)
  402f1f:	48 8b 42 10          	mov    0x10(%rdx),%rax
  402f23:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  402f27:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402f2b:	48 89 46 10          	mov    %rax,0x10(%rsi)
  402f2f:	48 8b 42 18          	mov    0x18(%rdx),%rax
  402f33:	48 89 4e 20          	mov    %rcx,0x20(%rsi)
  402f37:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  402f3b:	48 85 c0             	test   %rax,%rax
  402f3e:	48 89 46 18          	mov    %rax,0x18(%rsi)
  402f42:	48 89 4e 28          	mov    %rcx,0x28(%rsi)
  402f46:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  402f4a:	48 89 4e 30          	mov    %rcx,0x30(%rsi)
  402f4e:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  402f52:	c5 fa 7f 46 50       	vmovdqu %xmm0,0x50(%rsi)
  402f57:	48 89 4e 38          	mov    %rcx,0x38(%rsi)
  402f5b:	48 8d 4e 08          	lea    0x8(%rsi),%rcx
  402f5f:	48 89 4e 40          	mov    %rcx,0x40(%rsi)
  402f63:	48 8d 4e 50          	lea    0x50(%rsi),%rcx
  402f67:	48 89 4e 48          	mov    %rcx,0x48(%rsi)
  402f6b:	74 04                	je     402f71 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x531>
  402f6d:	f0 83 00 01          	lock addl $0x1,(%rax)
  402f71:	83 7a 04 02          	cmpl   $0x2,0x4(%rdx)
  402f75:	0f 8f 0f 03 00 00    	jg     40328a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x84a>
  402f7b:	48 8b 4a 48          	mov    0x48(%rdx),%rcx
  402f7f:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  402f84:	48 8b 09             	mov    (%rcx),%rcx
  402f87:	48 8b 46 48          	mov    0x48(%rsi),%rax
  402f8b:	48 89 08             	mov    %rcx,(%rax)
  402f8e:	48 8b 52 48          	mov    0x48(%rdx),%rdx
  402f92:	48 8b 46 48          	mov    0x48(%rsi),%rax
  402f96:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402f9a:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402f9e:	4d 8b 27             	mov    (%r15),%r12
  402fa1:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  402fa6:	4d 39 e6             	cmp    %r12,%r14
  402fa9:	75 36                	jne    402fe1 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x5a1>
  402fab:	e9 16 01 00 00       	jmpq   4030c6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  402fb0:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  402fb5:	48 8b 43 48          	mov    0x48(%rbx),%rax
  402fb9:	48 8b 12             	mov    (%rdx),%rdx
  402fbc:	48 89 10             	mov    %rdx,(%rax)
  402fbf:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  402fc4:	48 8b 43 48          	mov    0x48(%rbx),%rax
  402fc8:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402fcc:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402fd0:	49 83 c4 60          	add    $0x60,%r12
  402fd4:	48 83 c3 60          	add    $0x60,%rbx
  402fd8:	4d 39 e6             	cmp    %r12,%r14
  402fdb:	0f 84 e5 00 00 00    	je     4030c6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  402fe1:	48 85 db             	test   %rbx,%rbx
  402fe4:	74 ea                	je     402fd0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  402fe6:	41 8b 04 24          	mov    (%r12),%eax
  402fea:	89 03                	mov    %eax,(%rbx)
  402fec:	41 8b 44 24 04       	mov    0x4(%r12),%eax
  402ff1:	89 43 04             	mov    %eax,0x4(%rbx)
  402ff4:	41 8b 44 24 08       	mov    0x8(%r12),%eax
  402ff9:	89 43 08             	mov    %eax,0x8(%rbx)
  402ffc:	41 8b 44 24 0c       	mov    0xc(%r12),%eax
  403001:	89 43 0c             	mov    %eax,0xc(%rbx)
  403004:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  403009:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40300d:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  403012:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403016:	49 8b 54 24 20       	mov    0x20(%r12),%rdx
  40301b:	48 85 c0             	test   %rax,%rax
  40301e:	48 89 53 20          	mov    %rdx,0x20(%rbx)
  403022:	49 8b 54 24 28       	mov    0x28(%r12),%rdx
  403027:	48 89 53 28          	mov    %rdx,0x28(%rbx)
  40302b:	49 8b 54 24 30       	mov    0x30(%r12),%rdx
  403030:	48 89 53 30          	mov    %rdx,0x30(%rbx)
  403034:	49 8b 54 24 38       	mov    0x38(%r12),%rdx
  403039:	48 c7 43 58 00 00 00 	movq   $0x0,0x58(%rbx)
  403040:	00 
  403041:	48 c7 43 50 00 00 00 	movq   $0x0,0x50(%rbx)
  403048:	00 
  403049:	48 89 53 38          	mov    %rdx,0x38(%rbx)
  40304d:	48 8d 53 08          	lea    0x8(%rbx),%rdx
  403051:	48 89 53 40          	mov    %rdx,0x40(%rbx)
  403055:	48 8d 53 50          	lea    0x50(%rbx),%rdx
  403059:	48 89 53 48          	mov    %rdx,0x48(%rbx)
  40305d:	74 04                	je     403063 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x623>
  40305f:	f0 83 00 01          	lock addl $0x1,(%rax)
  403063:	41 83 7c 24 04 02    	cmpl   $0x2,0x4(%r12)
  403069:	0f 8e 41 ff ff ff    	jle    402fb0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x570>
  40306f:	c7 43 04 00 00 00 00 	movl   $0x0,0x4(%rbx)
  403076:	4c 89 e6             	mov    %r12,%rsi
  403079:	48 89 df             	mov    %rbx,%rdi
  40307c:	e8 7f e0 ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403081:	e9 4a ff ff ff       	jmpq   402fd0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  403086:	48 8d 34 3f          	lea    (%rdi,%rdi,1),%rsi
  40308a:	4c 89 f3             	mov    %r14,%rbx
  40308d:	48 29 c3             	sub    %rax,%rbx
  403090:	48 c1 fb 05          	sar    $0x5,%rbx
  403094:	48 39 f7             	cmp    %rsi,%rdi
  403097:	0f 86 79 02 00 00    	jbe    403316 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d6>
  40309d:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  4030a4:	aa aa aa 
  4030a7:	48 c7 c7 c0 ff ff ff 	mov    $0xffffffffffffffc0,%rdi
  4030ae:	48 0f af d8          	imul   %rax,%rbx
  4030b2:	48 b8 aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rax
  4030b9:	aa aa 02 
  4030bc:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  4030c1:	e9 22 fe ff ff       	jmpq   402ee8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4a8>
  4030c6:	49 8b 77 08          	mov    0x8(%r15),%rsi
  4030ca:	48 8d 43 60          	lea    0x60(%rbx),%rax
  4030ce:	4d 89 f5             	mov    %r14,%r13
  4030d1:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  4030d6:	49 89 c4             	mov    %rax,%r12
  4030d9:	49 39 f6             	cmp    %rsi,%r14
  4030dc:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  4030e1:	75 40                	jne    403123 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6e3>
  4030e3:	e9 fb 00 00 00       	jmpq   4031e3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7a3>
  4030e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4030ef:	00 
  4030f0:	49 8b 55 48          	mov    0x48(%r13),%rdx
  4030f4:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  4030f9:	48 8b 12             	mov    (%rdx),%rdx
  4030fc:	48 89 10             	mov    %rdx,(%rax)
  4030ff:	49 8b 55 48          	mov    0x48(%r13),%rdx
  403103:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  403108:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40310c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403110:	49 83 c4 60          	add    $0x60,%r12
  403114:	49 83 c5 60          	add    $0x60,%r13
  403118:	4c 39 6c 24 20       	cmp    %r13,0x20(%rsp)
  40311d:	0f 84 bc 00 00 00    	je     4031df <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x79f>
  403123:	4d 85 e4             	test   %r12,%r12
  403126:	74 e8                	je     403110 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  403128:	41 8b 45 00          	mov    0x0(%r13),%eax
  40312c:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  403131:	49 8d 4c 24 08       	lea    0x8(%r12),%rcx
  403136:	41 89 04 24          	mov    %eax,(%r12)
  40313a:	41 8b 45 04          	mov    0x4(%r13),%eax
  40313e:	41 89 44 24 04       	mov    %eax,0x4(%r12)
  403143:	41 8b 45 08          	mov    0x8(%r13),%eax
  403147:	41 89 44 24 08       	mov    %eax,0x8(%r12)
  40314c:	41 8b 45 0c          	mov    0xc(%r13),%eax
  403150:	41 89 44 24 0c       	mov    %eax,0xc(%r12)
  403155:	49 8b 45 10          	mov    0x10(%r13),%rax
  403159:	49 89 44 24 10       	mov    %rax,0x10(%r12)
  40315e:	49 8b 45 18          	mov    0x18(%r13),%rax
  403162:	49 89 44 24 18       	mov    %rax,0x18(%r12)
  403167:	49 8b 55 20          	mov    0x20(%r13),%rdx
  40316b:	49 89 54 24 20       	mov    %rdx,0x20(%r12)
  403170:	49 8b 55 28          	mov    0x28(%r13),%rdx
  403174:	49 89 54 24 28       	mov    %rdx,0x28(%r12)
  403179:	49 8b 55 30          	mov    0x30(%r13),%rdx
  40317d:	49 89 54 24 30       	mov    %rdx,0x30(%r12)
  403182:	49 8b 55 38          	mov    0x38(%r13),%rdx
  403186:	49 89 54 24 38       	mov    %rdx,0x38(%r12)
  40318b:	4c 89 e2             	mov    %r12,%rdx
  40318e:	48 29 da             	sub    %rbx,%rdx
  403191:	48 85 c0             	test   %rax,%rax
  403194:	48 89 4c 16 e0       	mov    %rcx,-0x20(%rsi,%rdx,1)
  403199:	49 8d 4c 24 50       	lea    0x50(%r12),%rcx
  40319e:	48 89 4c 16 e8       	mov    %rcx,-0x18(%rsi,%rdx,1)
  4031a3:	49 c7 44 24 58 00 00 	movq   $0x0,0x58(%r12)
  4031aa:	00 00 
  4031ac:	49 c7 44 24 50 00 00 	movq   $0x0,0x50(%r12)
  4031b3:	00 00 
  4031b5:	74 04                	je     4031bb <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x77b>
  4031b7:	f0 83 00 01          	lock addl $0x1,(%rax)
  4031bb:	41 83 7d 04 02       	cmpl   $0x2,0x4(%r13)
  4031c0:	0f 8e 2a ff ff ff    	jle    4030f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6b0>
  4031c6:	41 c7 44 24 04 00 00 	movl   $0x0,0x4(%r12)
  4031cd:	00 00 
  4031cf:	4c 89 ee             	mov    %r13,%rsi
  4031d2:	4c 89 e7             	mov    %r12,%rdi
  4031d5:	e8 26 df ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4031da:	e9 31 ff ff ff       	jmpq   403110 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  4031df:	4d 8b 77 08          	mov    0x8(%r15),%r14
  4031e3:	49 8b 1f             	mov    (%r15),%rbx
  4031e6:	4c 89 f7             	mov    %r14,%rdi
  4031e9:	4c 39 f3             	cmp    %r14,%rbx
  4031ec:	75 4f                	jne    40323d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7fd>
  4031ee:	eb 69                	jmp    403259 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x819>
  4031f0:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4031f4:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4031fb:	00 
  4031fc:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403203:	00 
  403204:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40320b:	00 
  40320c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403213:	00 
  403214:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40321a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40321e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403222:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403229:	00 
  40322a:	48 39 c7             	cmp    %rax,%rdi
  40322d:	74 05                	je     403234 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7f4>
  40322f:	e8 0c e0 ff ff       	callq  401240 <_ZN2cv8fastFreeEPv@plt>
  403234:	48 83 c3 60          	add    $0x60,%rbx
  403238:	4c 39 f3             	cmp    %r14,%rbx
  40323b:	74 19                	je     403256 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x816>
  40323d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403241:	48 85 c0             	test   %rax,%rax
  403244:	74 aa                	je     4031f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403246:	f0 83 28 01          	lock subl $0x1,(%rax)
  40324a:	75 a4                	jne    4031f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  40324c:	48 89 df             	mov    %rbx,%rdi
  40324f:	e8 0c df ff ff       	callq  401160 <_ZN2cv3Mat10deallocateEv@plt>
  403254:	eb 9a                	jmp    4031f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403256:	49 8b 3f             	mov    (%r15),%rdi
  403259:	48 85 ff             	test   %rdi,%rdi
  40325c:	74 07                	je     403265 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x825>
  40325e:	66 90                	xchg   %ax,%ax
  403260:	e8 0b df ff ff       	callq  401170 <_ZdlPv@plt>
  403265:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40326a:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  40326f:	4d 89 67 08          	mov    %r12,0x8(%r15)
  403273:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403277:	49 89 37             	mov    %rsi,(%r15)
  40327a:	48 c1 e0 05          	shl    $0x5,%rax
  40327e:	48 01 f0             	add    %rsi,%rax
  403281:	49 89 47 10          	mov    %rax,0x10(%r15)
  403285:	e9 5b fb ff ff       	jmpq   402de5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  40328a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40328f:	48 89 d6             	mov    %rdx,%rsi
  403292:	c7 40 04 00 00 00 00 	movl   $0x0,0x4(%rax)
  403299:	48 89 c7             	mov    %rax,%rdi
  40329c:	e8 5f de ff ff       	callq  401100 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4032a1:	e9 f8 fc ff ff       	jmpq   402f9e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  4032a6:	48 89 c7             	mov    %rax,%rdi
  4032a9:	c5 f8 77             	vzeroupper 
  4032ac:	e8 2f de ff ff       	callq  4010e0 <__cxa_begin_catch@plt>
  4032b1:	4c 39 64 24 28       	cmp    %r12,0x28(%rsp)
  4032b6:	74 16                	je     4032ce <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x88e>
  4032b8:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
  4032bd:	48 89 df             	mov    %rbx,%rdi
  4032c0:	e8 1b f6 ff ff       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4032c5:	48 83 c3 60          	add    $0x60,%rbx
  4032c9:	4c 39 e3             	cmp    %r12,%rbx
  4032cc:	75 ef                	jne    4032bd <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x87d>
  4032ce:	e8 3d df ff ff       	callq  401210 <__cxa_rethrow@plt>
  4032d3:	48 89 c3             	mov    %rax,%rbx
  4032d6:	4c 89 ef             	mov    %r13,%rdi
  4032d9:	c5 f8 77             	vzeroupper 
  4032dc:	e8 ff f5 ff ff       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4032e1:	48 89 df             	mov    %rbx,%rdi
  4032e4:	e8 c7 de ff ff       	callq  4011b0 <_Unwind_Resume@plt>
  4032e9:	48 89 c7             	mov    %rax,%rdi
  4032ec:	c5 f8 77             	vzeroupper 
  4032ef:	e8 ec dd ff ff       	callq  4010e0 <__cxa_begin_catch@plt>
  4032f4:	48 39 5c 24 18       	cmp    %rbx,0x18(%rsp)
  4032f9:	74 16                	je     403311 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d1>
  4032fb:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  403300:	4c 89 e7             	mov    %r12,%rdi
  403303:	e8 d8 f5 ff ff       	callq  4028e0 <_ZN2cv3MatD1Ev>
  403308:	49 83 c4 60          	add    $0x60,%r12
  40330c:	4c 39 e3             	cmp    %r12,%rbx
  40330f:	75 ef                	jne    403300 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8c0>
  403311:	e8 fa de ff ff       	callq  401210 <__cxa_rethrow@plt>
  403316:	48 bf aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rdi
  40331d:	aa aa 02 
  403320:	48 39 fe             	cmp    %rdi,%rsi
  403323:	0f 87 74 fd ff ff    	ja     40309d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x65d>
  403329:	48 0f af d9          	imul   %rcx,%rbx
  40332d:	48 85 f6             	test   %rsi,%rsi
  403330:	0f 85 a5 fb ff ff    	jne    402edb <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x49b>
  403336:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40333d:	00 00 
  40333f:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  403346:	00 00 
  403348:	e9 af fb ff ff       	jmpq   402efc <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4bc>
  40334d:	48 89 c7             	mov    %rax,%rdi
  403350:	c5 f8 77             	vzeroupper 
  403353:	e8 88 dd ff ff       	callq  4010e0 <__cxa_begin_catch@plt>
  403358:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  40335e:	74 69                	je     4033c9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  403360:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  403365:	e8 06 de ff ff       	callq  401170 <_ZdlPv@plt>
  40336a:	e8 a1 de ff ff       	callq  401210 <__cxa_rethrow@plt>
  40336f:	48 89 c3             	mov    %rax,%rbx
  403372:	c5 f8 77             	vzeroupper 
  403375:	e8 76 de ff ff       	callq  4011f0 <__cxa_end_catch@plt>
  40337a:	48 89 df             	mov    %rbx,%rdi
  40337d:	e8 5e dd ff ff       	callq  4010e0 <__cxa_begin_catch@plt>
  403382:	48 83 7c 24 28 00    	cmpq   $0x0,0x28(%rsp)
  403388:	74 3f                	je     4033c9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  40338a:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40338f:	48 39 44 24 28       	cmp    %rax,0x28(%rsp)
  403394:	74 ca                	je     403360 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  403396:	48 89 c3             	mov    %rax,%rbx
  403399:	48 89 df             	mov    %rbx,%rdi
  40339c:	e8 3f f5 ff ff       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4033a1:	48 83 c3 60          	add    $0x60,%rbx
  4033a5:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
  4033aa:	75 ed                	jne    403399 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x959>
  4033ac:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  4033b2:	75 ac                	jne    403360 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  4033b4:	eb b4                	jmp    40336a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x92a>
  4033b6:	48 89 c3             	mov    %rax,%rbx
  4033b9:	c5 f8 77             	vzeroupper 
  4033bc:	e8 2f de ff ff       	callq  4011f0 <__cxa_end_catch@plt>
  4033c1:	48 89 df             	mov    %rbx,%rdi
  4033c4:	e8 17 dd ff ff       	callq  4010e0 <__cxa_begin_catch@plt>
  4033c9:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4033ce:	e8 0d f5 ff ff       	callq  4028e0 <_ZN2cv3MatD1Ev>
  4033d3:	eb d7                	jmp    4033ac <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x96c>
  4033d5:	48 89 c3             	mov    %rax,%rbx
  4033d8:	c5 f8 77             	vzeroupper 
  4033db:	e8 10 de ff ff       	callq  4011f0 <__cxa_end_catch@plt>
  4033e0:	48 89 df             	mov    %rbx,%rdi
  4033e3:	e8 c8 dd ff ff       	callq  4011b0 <_Unwind_Resume@plt>
  4033e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4033ef:	00 

00000000004033f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>:
  4033f0:	55                   	push   %rbp
  4033f1:	48 89 e5             	mov    %rsp,%rbp
  4033f4:	41 56                	push   %r14
  4033f6:	41 55                	push   %r13
  4033f8:	41 54                	push   %r12
  4033fa:	49 89 d4             	mov    %rdx,%r12
  4033fd:	49 29 f4             	sub    %rsi,%r12
  403400:	53                   	push   %rbx
  403401:	48 89 fb             	mov    %rdi,%rbx
  403404:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  403408:	48 83 ec 10          	sub    $0x10,%rsp
  40340c:	48 8b 07             	mov    (%rdi),%rax
  40340f:	48 8b 4f 10          	mov    0x10(%rdi),%rcx
  403413:	48 29 c1             	sub    %rax,%rcx
  403416:	48 c1 f9 02          	sar    $0x2,%rcx
  40341a:	49 39 cc             	cmp    %rcx,%r12
  40341d:	0f 87 0d 04 00 00    	ja     403830 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x440>
  403423:	4c 8b 4f 08          	mov    0x8(%rdi),%r9
  403427:	4c 89 cf             	mov    %r9,%rdi
  40342a:	48 29 c7             	sub    %rax,%rdi
  40342d:	48 c1 ff 02          	sar    $0x2,%rdi
  403431:	49 39 fc             	cmp    %rdi,%r12
  403434:	0f 87 2e 01 00 00    	ja     403568 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x178>
  40343a:	4d 85 e4             	test   %r12,%r12
  40343d:	0f 8e a5 05 00 00    	jle    4039e8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5f8>
  403443:	4a 8d 3c a0          	lea    (%rax,%r12,4),%rdi
  403447:	4a 8d 0c 26          	lea    (%rsi,%r12,1),%rcx
  40344b:	48 39 fe             	cmp    %rdi,%rsi
  40344e:	41 0f 93 c0          	setae  %r8b
  403452:	48 39 c8             	cmp    %rcx,%rax
  403455:	0f 93 c2             	setae  %dl
  403458:	41 08 d0             	or     %dl,%r8b
  40345b:	0f 84 8f 03 00 00    	je     4037f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403461:	49 83 fc 1f          	cmp    $0x1f,%r12
  403465:	0f 86 85 03 00 00    	jbe    4037f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  40346b:	4d 89 e0             	mov    %r12,%r8
  40346e:	48 8d 50 20          	lea    0x20(%rax),%rdx
  403472:	45 31 c9             	xor    %r9d,%r9d
  403475:	49 c1 e8 05          	shr    $0x5,%r8
  403479:	4c 89 c1             	mov    %r8,%rcx
  40347c:	48 c1 e1 05          	shl    $0x5,%rcx
  403480:	4d 89 ca             	mov    %r9,%r10
  403483:	49 83 c1 01          	add    $0x1,%r9
  403487:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  40348b:	49 c1 e2 05          	shl    $0x5,%r10
  40348f:	c4 a1 7a 6f 04 16    	vmovdqu (%rsi,%r10,1),%xmm0
  403495:	c4 a3 7d 38 44 16 10 	vinserti128 $0x1,0x10(%rsi,%r10,1),%ymm0,%ymm0
  40349c:	01 
  40349d:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  4034a2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  4034a8:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  4034ad:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  4034b3:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  4034b8:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  4034bc:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  4034c3:	ff 
  4034c4:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  4034c9:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  4034d0:	ff ff 01 
  4034d3:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  4034d7:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  4034dc:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  4034e3:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  4034e8:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  4034ee:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  4034f2:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  4034f7:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  4034fe:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403503:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403507:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  40350c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  403513:	4d 39 c1             	cmp    %r8,%r9
  403516:	0f 82 64 ff ff ff    	jb     403480 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x90>
  40351c:	48 8d 14 88          	lea    (%rax,%rcx,4),%rdx
  403520:	4c 89 e0             	mov    %r12,%rax
  403523:	48 01 ce             	add    %rcx,%rsi
  403526:	48 29 c8             	sub    %rcx,%rax
  403529:	49 39 cc             	cmp    %rcx,%r12
  40352c:	74 1f                	je     40354d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x15d>
  40352e:	66 90                	xchg   %ax,%ax
  403530:	0f b6 0e             	movzbl (%rsi),%ecx
  403533:	48 83 e8 01          	sub    $0x1,%rax
  403537:	48 83 c6 01          	add    $0x1,%rsi
  40353b:	48 83 c2 04          	add    $0x4,%rdx
  40353f:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  403543:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  403548:	48 85 c0             	test   %rax,%rax
  40354b:	7f e3                	jg     403530 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x140>
  40354d:	c5 f8 77             	vzeroupper 
  403550:	48 89 7b 08          	mov    %rdi,0x8(%rbx)
  403554:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  403558:	5b                   	pop    %rbx
  403559:	41 5c                	pop    %r12
  40355b:	41 5d                	pop    %r13
  40355d:	41 5e                	pop    %r14
  40355f:	5d                   	pop    %rbp
  403560:	c3                   	retq   
  403561:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403568:	48 01 f7             	add    %rsi,%rdi
  40356b:	49 89 fa             	mov    %rdi,%r10
  40356e:	49 29 f2             	sub    %rsi,%r10
  403571:	4d 85 d2             	test   %r10,%r10
  403574:	0f 8e 14 01 00 00    	jle    40368e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  40357a:	4a 8d 0c 90          	lea    (%rax,%r10,4),%rcx
  40357e:	4e 8d 04 16          	lea    (%rsi,%r10,1),%r8
  403582:	48 39 ce             	cmp    %rcx,%rsi
  403585:	41 0f 93 c3          	setae  %r11b
  403589:	4c 39 c0             	cmp    %r8,%rax
  40358c:	0f 93 c1             	setae  %cl
  40358f:	41 08 cb             	or     %cl,%r11b
  403592:	0f 84 30 02 00 00    	je     4037c8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403598:	49 83 fa 1f          	cmp    $0x1f,%r10
  40359c:	0f 86 26 02 00 00    	jbe    4037c8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  4035a2:	4d 89 d0             	mov    %r10,%r8
  4035a5:	48 8d 48 20          	lea    0x20(%rax),%rcx
  4035a9:	45 31 e4             	xor    %r12d,%r12d
  4035ac:	49 c1 e8 05          	shr    $0x5,%r8
  4035b0:	4d 89 c3             	mov    %r8,%r11
  4035b3:	49 c1 e3 05          	shl    $0x5,%r11
  4035b7:	4d 89 e5             	mov    %r12,%r13
  4035ba:	49 83 c4 01          	add    $0x1,%r12
  4035be:	48 83 e9 80          	sub    $0xffffffffffffff80,%rcx
  4035c2:	49 c1 e5 05          	shl    $0x5,%r13
  4035c6:	c4 a1 7a 6f 04 2e    	vmovdqu (%rsi,%r13,1),%xmm0
  4035cc:	c4 a3 7d 38 44 2e 10 	vinserti128 $0x1,0x10(%rsi,%r13,1),%ymm0,%ymm0
  4035d3:	01 
  4035d4:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  4035d9:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  4035df:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  4035e4:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  4035ea:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  4035ef:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  4035f3:	c5 f8 11 91 60 ff ff 	vmovups %xmm2,-0xa0(%rcx)
  4035fa:	ff 
  4035fb:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403600:	c4 e3 7d 19 91 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rcx)
  403607:	ff ff 01 
  40360a:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  40360e:	c5 f8 11 49 80       	vmovups %xmm1,-0x80(%rcx)
  403613:	c4 e3 7d 19 49 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rcx)
  40361a:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  40361f:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403625:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403629:	c5 f8 11 49 a0       	vmovups %xmm1,-0x60(%rcx)
  40362e:	c4 e3 7d 19 49 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rcx)
  403635:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  40363a:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  40363e:	c5 f8 11 41 c0       	vmovups %xmm0,-0x40(%rcx)
  403643:	c4 e3 7d 19 41 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rcx)
  40364a:	4d 39 c4             	cmp    %r8,%r12
  40364d:	0f 82 64 ff ff ff    	jb     4035b7 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1c7>
  403653:	4a 8d 0c 98          	lea    (%rax,%r11,4),%rcx
  403657:	4c 89 d0             	mov    %r10,%rax
  40365a:	4e 8d 04 1e          	lea    (%rsi,%r11,1),%r8
  40365e:	4c 29 d8             	sub    %r11,%rax
  403661:	4d 39 da             	cmp    %r11,%r10
  403664:	74 28                	je     40368e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403666:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40366d:	00 00 00 
  403670:	41 0f b6 30          	movzbl (%r8),%esi
  403674:	48 83 e8 01          	sub    $0x1,%rax
  403678:	49 83 c0 01          	add    $0x1,%r8
  40367c:	48 83 c1 04          	add    $0x4,%rcx
  403680:	c5 fa 2a c6          	vcvtsi2ss %esi,%xmm0,%xmm0
  403684:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  403689:	48 85 c0             	test   %rax,%rax
  40368c:	7f e2                	jg     403670 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x280>
  40368e:	48 29 fa             	sub    %rdi,%rdx
  403691:	4d 89 c8             	mov    %r9,%r8
  403694:	48 85 d2             	test   %rdx,%rdx
  403697:	0f 8e 10 01 00 00    	jle    4037ad <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  40369d:	4d 8d 04 91          	lea    (%r9,%rdx,4),%r8
  4036a1:	48 8d 0c 17          	lea    (%rdi,%rdx,1),%rcx
  4036a5:	4c 39 c7             	cmp    %r8,%rdi
  4036a8:	40 0f 93 c6          	setae  %sil
  4036ac:	49 39 c9             	cmp    %rcx,%r9
  4036af:	0f 93 c0             	setae  %al
  4036b2:	40 08 c6             	or     %al,%sil
  4036b5:	0f 84 55 01 00 00    	je     403810 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  4036bb:	48 83 fa 1f          	cmp    $0x1f,%rdx
  4036bf:	0f 86 4b 01 00 00    	jbe    403810 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  4036c5:	48 89 d1             	mov    %rdx,%rcx
  4036c8:	49 8d 41 20          	lea    0x20(%r9),%rax
  4036cc:	31 f6                	xor    %esi,%esi
  4036ce:	48 c1 e9 05          	shr    $0x5,%rcx
  4036d2:	49 89 ca             	mov    %rcx,%r10
  4036d5:	49 c1 e2 05          	shl    $0x5,%r10
  4036d9:	49 89 f3             	mov    %rsi,%r11
  4036dc:	48 83 c6 01          	add    $0x1,%rsi
  4036e0:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  4036e4:	49 c1 e3 05          	shl    $0x5,%r11
  4036e8:	c4 a1 7a 6f 04 1f    	vmovdqu (%rdi,%r11,1),%xmm0
  4036ee:	c4 a3 7d 38 44 1f 10 	vinserti128 $0x1,0x10(%rdi,%r11,1),%ymm0,%ymm0
  4036f5:	01 
  4036f6:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  4036fb:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403701:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403706:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  40370c:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403711:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403715:	c5 f8 11 90 60 ff ff 	vmovups %xmm2,-0xa0(%rax)
  40371c:	ff 
  40371d:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403722:	c4 e3 7d 19 90 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rax)
  403729:	ff ff 01 
  40372c:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403730:	c5 f8 11 48 80       	vmovups %xmm1,-0x80(%rax)
  403735:	c4 e3 7d 19 48 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rax)
  40373c:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403741:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403747:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  40374b:	c5 f8 11 48 a0       	vmovups %xmm1,-0x60(%rax)
  403750:	c4 e3 7d 19 48 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rax)
  403757:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  40375c:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403760:	c5 f8 11 40 c0       	vmovups %xmm0,-0x40(%rax)
  403765:	c4 e3 7d 19 40 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rax)
  40376c:	48 39 ce             	cmp    %rcx,%rsi
  40376f:	0f 82 64 ff ff ff    	jb     4036d9 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x2e9>
  403775:	48 89 d0             	mov    %rdx,%rax
  403778:	4a 8d 34 17          	lea    (%rdi,%r10,1),%rsi
  40377c:	4b 8d 0c 91          	lea    (%r9,%r10,4),%rcx
  403780:	4c 29 d0             	sub    %r10,%rax
  403783:	4c 39 d2             	cmp    %r10,%rdx
  403786:	74 25                	je     4037ad <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  403788:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40378f:	00 
  403790:	0f b6 16             	movzbl (%rsi),%edx
  403793:	48 83 e8 01          	sub    $0x1,%rax
  403797:	48 83 c6 01          	add    $0x1,%rsi
  40379b:	48 83 c1 04          	add    $0x4,%rcx
  40379f:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  4037a3:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  4037a8:	48 85 c0             	test   %rax,%rax
  4037ab:	7f e3                	jg     403790 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3a0>
  4037ad:	4c 89 43 08          	mov    %r8,0x8(%rbx)
  4037b1:	c5 f8 77             	vzeroupper 
  4037b4:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  4037b8:	5b                   	pop    %rbx
  4037b9:	41 5c                	pop    %r12
  4037bb:	41 5d                	pop    %r13
  4037bd:	41 5e                	pop    %r14
  4037bf:	5d                   	pop    %rbp
  4037c0:	c3                   	retq   
  4037c1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4037c8:	0f b6 0e             	movzbl (%rsi),%ecx
  4037cb:	48 83 c6 01          	add    $0x1,%rsi
  4037cf:	48 83 c0 04          	add    $0x4,%rax
  4037d3:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  4037d7:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  4037dc:	49 39 f0             	cmp    %rsi,%r8
  4037df:	75 e7                	jne    4037c8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  4037e1:	e9 a8 fe ff ff       	jmpq   40368e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  4037e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4037ed:	00 00 00 
  4037f0:	0f b6 16             	movzbl (%rsi),%edx
  4037f3:	48 83 c6 01          	add    $0x1,%rsi
  4037f7:	48 83 c0 04          	add    $0x4,%rax
  4037fb:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  4037ff:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  403804:	48 39 f1             	cmp    %rsi,%rcx
  403807:	75 e7                	jne    4037f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403809:	e9 42 fd ff ff       	jmpq   403550 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  40380e:	66 90                	xchg   %ax,%ax
  403810:	0f b6 07             	movzbl (%rdi),%eax
  403813:	48 83 c7 01          	add    $0x1,%rdi
  403817:	49 83 c1 04          	add    $0x4,%r9
  40381b:	c5 fa 2a c0          	vcvtsi2ss %eax,%xmm0,%xmm0
  40381f:	c4 c1 7a 11 41 fc    	vmovss %xmm0,-0x4(%r9)
  403825:	48 39 f9             	cmp    %rdi,%rcx
  403828:	75 e6                	jne    403810 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  40382a:	eb 81                	jmp    4037ad <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  40382c:	0f 1f 40 00          	nopl   0x0(%rax)
  403830:	4d 85 e4             	test   %r12,%r12
  403833:	0f 84 9f 01 00 00    	je     4039d8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5e8>
  403839:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
  403840:	ff ff 3f 
  403843:	49 39 c4             	cmp    %rax,%r12
  403846:	0f 87 a4 01 00 00    	ja     4039f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x600>
  40384c:	4e 8d 34 a5 00 00 00 	lea    0x0(,%r12,4),%r14
  403853:	00 
  403854:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  403859:	4c 89 f7             	mov    %r14,%rdi
  40385c:	e8 bf d8 ff ff       	callq  401120 <_Znwm@plt>
  403861:	4d 85 e4             	test   %r12,%r12
  403864:	49 89 c5             	mov    %rax,%r13
  403867:	48 8b 3b             	mov    (%rbx),%rdi
  40386a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40386f:	0f 8e 0d 01 00 00    	jle    403982 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  403875:	4a 8d 04 30          	lea    (%rax,%r14,1),%rax
  403879:	4e 8d 04 26          	lea    (%rsi,%r12,1),%r8
  40387d:	48 39 c6             	cmp    %rax,%rsi
  403880:	0f 93 c2             	setae  %dl
  403883:	4d 39 e8             	cmp    %r13,%r8
  403886:	0f 96 c0             	setbe  %al
  403889:	08 c2                	or     %al,%dl
  40388b:	0f 84 1f 01 00 00    	je     4039b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  403891:	49 83 fc 1f          	cmp    $0x1f,%r12
  403895:	0f 86 15 01 00 00    	jbe    4039b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  40389b:	4c 89 e1             	mov    %r12,%rcx
  40389e:	49 8d 55 20          	lea    0x20(%r13),%rdx
  4038a2:	45 31 c0             	xor    %r8d,%r8d
  4038a5:	48 c1 e9 05          	shr    $0x5,%rcx
  4038a9:	48 89 c8             	mov    %rcx,%rax
  4038ac:	48 c1 e0 05          	shl    $0x5,%rax
  4038b0:	4d 89 c1             	mov    %r8,%r9
  4038b3:	49 83 c0 01          	add    $0x1,%r8
  4038b7:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  4038bb:	49 c1 e1 05          	shl    $0x5,%r9
  4038bf:	c4 a1 7a 6f 04 0e    	vmovdqu (%rsi,%r9,1),%xmm0
  4038c5:	c4 a3 7d 38 44 0e 10 	vinserti128 $0x1,0x10(%rsi,%r9,1),%ymm0,%ymm0
  4038cc:	01 
  4038cd:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  4038d2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  4038d8:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  4038dd:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  4038e3:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  4038e8:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  4038ec:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  4038f3:	ff 
  4038f4:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  4038f9:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  403900:	ff ff 01 
  403903:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403907:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  40390c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  403913:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403918:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40391e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403922:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  403927:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  40392e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403933:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403937:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  40393c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  403943:	4c 39 c1             	cmp    %r8,%rcx
  403946:	0f 87 64 ff ff ff    	ja     4038b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x4c0>
  40394c:	4c 89 e2             	mov    %r12,%rdx
  40394f:	48 01 c6             	add    %rax,%rsi
  403952:	49 8d 4c 85 00       	lea    0x0(%r13,%rax,4),%rcx
  403957:	48 29 c2             	sub    %rax,%rdx
  40395a:	49 39 c4             	cmp    %rax,%r12
  40395d:	74 20                	je     40397f <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x58f>
  40395f:	90                   	nop
  403960:	44 0f b6 06          	movzbl (%rsi),%r8d
  403964:	48 83 ea 01          	sub    $0x1,%rdx
  403968:	48 83 c6 01          	add    $0x1,%rsi
  40396c:	48 83 c1 04          	add    $0x4,%rcx
  403970:	c4 c1 7a 2a c0       	vcvtsi2ss %r8d,%xmm0,%xmm0
  403975:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  40397a:	48 85 d2             	test   %rdx,%rdx
  40397d:	7f e1                	jg     403960 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x570>
  40397f:	c5 f8 77             	vzeroupper 
  403982:	48 85 ff             	test   %rdi,%rdi
  403985:	74 05                	je     40398c <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x59c>
  403987:	e8 e4 d7 ff ff       	callq  401170 <_ZdlPv@plt>
  40398c:	4c 89 2b             	mov    %r13,(%rbx)
  40398f:	4d 01 f5             	add    %r14,%r13
  403992:	4c 89 6b 08          	mov    %r13,0x8(%rbx)
  403996:	4c 89 6b 10          	mov    %r13,0x10(%rbx)
  40399a:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  40399e:	5b                   	pop    %rbx
  40399f:	41 5c                	pop    %r12
  4039a1:	41 5d                	pop    %r13
  4039a3:	41 5e                	pop    %r14
  4039a5:	5d                   	pop    %rbp
  4039a6:	c3                   	retq   
  4039a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4039ae:	00 00 
  4039b0:	4c 89 ea             	mov    %r13,%rdx
  4039b3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4039b8:	0f b6 0e             	movzbl (%rsi),%ecx
  4039bb:	48 83 c6 01          	add    $0x1,%rsi
  4039bf:	48 83 c2 04          	add    $0x4,%rdx
  4039c3:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  4039c7:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  4039cc:	49 39 f0             	cmp    %rsi,%r8
  4039cf:	75 e7                	jne    4039b8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c8>
  4039d1:	eb af                	jmp    403982 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4039d3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4039d8:	48 89 c7             	mov    %rax,%rdi
  4039db:	45 31 f6             	xor    %r14d,%r14d
  4039de:	45 31 ed             	xor    %r13d,%r13d
  4039e1:	eb 9f                	jmp    403982 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4039e3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4039e8:	48 89 c7             	mov    %rax,%rdi
  4039eb:	e9 60 fb ff ff       	jmpq   403550 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  4039f0:	e8 1b d7 ff ff       	callq  401110 <_ZSt17__throw_bad_allocv@plt>
  4039f5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4039fc:	00 00 00 
  4039ff:	90                   	nop

0000000000403a00 <__libc_csu_init>:
  403a00:	41 57                	push   %r15
  403a02:	41 89 ff             	mov    %edi,%r15d
  403a05:	41 56                	push   %r14
  403a07:	49 89 f6             	mov    %rsi,%r14
  403a0a:	41 55                	push   %r13
  403a0c:	49 89 d5             	mov    %rdx,%r13
  403a0f:	41 54                	push   %r12
  403a11:	4c 8d 25 c0 12 20 00 	lea    0x2012c0(%rip),%r12        # 604cd8 <__frame_dummy_init_array_entry>
  403a18:	55                   	push   %rbp
  403a19:	48 8d 2d c8 12 20 00 	lea    0x2012c8(%rip),%rbp        # 604ce8 <__init_array_end>
  403a20:	53                   	push   %rbx
  403a21:	4c 29 e5             	sub    %r12,%rbp
  403a24:	31 db                	xor    %ebx,%ebx
  403a26:	48 c1 fd 03          	sar    $0x3,%rbp
  403a2a:	48 83 ec 08          	sub    $0x8,%rsp
  403a2e:	e8 7d d6 ff ff       	callq  4010b0 <_init>
  403a33:	48 85 ed             	test   %rbp,%rbp
  403a36:	74 1e                	je     403a56 <__libc_csu_init+0x56>
  403a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403a3f:	00 
  403a40:	4c 89 ea             	mov    %r13,%rdx
  403a43:	4c 89 f6             	mov    %r14,%rsi
  403a46:	44 89 ff             	mov    %r15d,%edi
  403a49:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  403a4d:	48 83 c3 01          	add    $0x1,%rbx
  403a51:	48 39 eb             	cmp    %rbp,%rbx
  403a54:	75 ea                	jne    403a40 <__libc_csu_init+0x40>
  403a56:	48 83 c4 08          	add    $0x8,%rsp
  403a5a:	5b                   	pop    %rbx
  403a5b:	5d                   	pop    %rbp
  403a5c:	41 5c                	pop    %r12
  403a5e:	41 5d                	pop    %r13
  403a60:	41 5e                	pop    %r14
  403a62:	41 5f                	pop    %r15
  403a64:	c3                   	retq   
  403a65:	90                   	nop
  403a66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403a6d:	00 00 00 

0000000000403a70 <__libc_csu_fini>:
  403a70:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000403a74 <_fini>:
  403a74:	48 83 ec 08          	sub    $0x8,%rsp
  403a78:	48 83 c4 08          	add    $0x8,%rsp
  403a7c:	c3                   	retq   
