import ctypes as ct
import os
import argparse
import sys

# 공유 라이브러리 경로
lib_path = '../build/GOLdatagen.so'

lib = ct.CDLL(lib_path)

lib.genGOLdata.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
lib.genGOLdata.restype = None

lib.genGOLdataInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
lib.genGOLdataInOneFile.restype = None

def main():
    parser = argparse.ArgumentParser(description='Game of Life 데이터 생성기')
    parser.add_argument('param1', type=int, help='첫 번째 매개변수 (uint32)')
    parser.add_argument('param2', type=int, help='두 번째 매개변수 (uint32)')
    parser.add_argument('param3', type=float, help='세 번째 매개변수 (double)')
    
    # 선택적 인자들
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--output', '-o', type=str, help='출력 디렉토리')
    parser.add_argument('--one_file', type=bool, default=False, help='한 파일에 데이터를 몰아서 저장')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"매개변수: {args.param1}, {args.param2}, {args.param}")
        if args.output:
            print(f"출력 디렉토리: {args.output}")
    
    # C 라이브러리 함수 호출
    if args.one_file:
        try:
            lib.genGOLdataInOneFile(args.param1, args.param2, args.param3)
            print("데이터 생성 완료!")
        except Exception as e:
            print(f"오류 발생: {e}")
            return 1
    else:
        try:
            lib.genGOLdata(args.param1, args.param2, args.param3)
            print("데이터 생성 완료!")
        except Exception as e:
            print(f"오류 발생: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
