import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Демо-приложение с типичными проблемами производительности — на нём удобно
 * проверять профилировщик. Запуск без сборки: {@code java Busy.java}.
 *
 * <ul>
 *   <li>{@code hot-loop} — занимает CPU подсчётом простых чисел;</li>
 *   <li>{@code allocator} — непрерывно создаёт строки и массивы;</li>
 *   <li>{@code worker-1..3} — долго держат общий монитор и ждут друг друга.</li>
 * </ul>
 */
public class Busy {
    private static final Object SHARED_LOCK = new Object();
    private static volatile long sink;

    public static void main(String[] args) throws InterruptedException {
        start("hot-loop", Busy::hotLoop);
        start("allocator", Busy::allocate);
        for (int i = 1; i <= 3; i++) {
            start("worker-" + i, Busy::contend);
        }
        System.out.println("Busy запущен, pid " + ProcessHandle.current().pid() + ". Ctrl+C — выход.");
        Thread.currentThread().join();
    }

    private static void start(String name, Runnable body) {
        Thread thread = new Thread(body, name);
        thread.setDaemon(true);
        thread.start();
    }

    private static void hotLoop() {
        while (true) {
            sink += countPrimes(200_000);
        }
    }

    static int countPrimes(int limit) {
        int count = 0;
        for (int n = 2; n < limit; n++) {
            if (isPrime(n)) {
                count++;
            }
        }
        return count;
    }

    static boolean isPrime(int n) {
        for (int d = 2; (long) d * d <= n; d++) {
            if (n % d == 0) {
                return false;
            }
        }
        return true;
    }

    private static void allocate() {
        Deque<Object> retained = new ArrayDeque<>();
        long i = 0;
        while (true) {
            retained.addLast(buildReport(i++));
            retained.addLast(new byte[4096]);
            if (retained.size() > 20_000) {
                retained.pollFirst();
                retained.pollFirst();
            }
        }
    }

    static String buildReport(long id) {
        StringBuilder report = new StringBuilder();
        for (int line = 0; line < 20; line++) {
            report.append("report ").append(id).append(" line ").append(line).append('\n');
        }
        return report.toString();
    }

    private static void contend() {
        while (true) {
            synchronized (SHARED_LOCK) {
                sink += countPrimes(20_000);
            }
        }
    }
}
